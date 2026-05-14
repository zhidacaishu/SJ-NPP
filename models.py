import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SJNPP(nn.Module):
    """
    PyTorch implementation of SJ-NPP, the attention-based neural point process
    model introduced in the paper "When, How, and What: An Attention-Based
    Neural Point Process for Joint Modeling of E-commerce Event Sequences".

    Attributes:
        num_seq: number of user sequences
        num_type: number of action type
        nhid: hidden state size
        nlayers: number of layers for the hidden states
        emsize: embedding state size
        dropout: dropout rate
        encoder: encoder of the hidden states, either 'GRU' or 'LSTM'
        device: device used to host the model, either 'cpu' or 'gpu'
        item_emsize: embedding size of item
        num_topic: number of item topic
        w: task-specific parameters used in the multi-rep. attn. algo
        topic_emb: the embedding of topics 
        h2o_time_bias: user-specific baseline occurrence rate of purchase activities
        h2o_action_bias: user-specific bias terms for the parameters of action
        h2o_item_bias: user-specific bias terms for the parameters of item
    """
    def __init__(self, config):
        super(SJNPP, self).__init__()

        self.num_seq = config['num_seq']
        self.num_type = config['num_type']
        self.nhid = config['nhid']
        self.nlayers = config['nlayers']
        self.emsize = config['emsize']
        self.dropout = config['dropout']
        self.encoder = config['encoder']
        self.device = config['device']
        self.action_encoding = config['action_encoding']
        self.item_emsize = config['item_emsize']
        self.num_topic = config['num_topic']
        self.num_prod = config['num_prod']
        self.self_embedding = config['self_embedding']
        self.nhead = config['nhead']

        self.lr_time = nn.Sequential(
            nn.Linear(1, self.item_emsize),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.item_emsize, 3 * self.item_emsize),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(3 * self.item_emsize, self.item_emsize)
        )

        if self.action_encoding == 'one-hot':
            self.lr_action = nn.Sequential(
                nn.Linear(self.num_type, self.item_emsize),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.item_emsize, 3 * self.item_emsize),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(3 * self.item_emsize, self.item_emsize)
            )
        elif self.action_encoding == 'index':
            self.lr_action = nn.Sequential(
                nn.Linear(1, self.item_emsize),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.item_emsize, 3 * self.item_emsize),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(3 * self.item_emsize, self.item_emsize)
            )
        else:
            raise KeyError('Unsupported action encoding!!!')

        self.inter_feature_attention = nn.MultiheadAttention(embed_dim=self.item_emsize, num_heads=self.nhead, batch_first=True, dropout=self.dropout)
        self.fusion_mlp_after_attn = nn.Linear(3 * self.item_emsize, self.emsize)

        if self.encoder == 'lstm':
            self.rnn = nn.LSTM(self.emsize, self.nhid, self.nlayers, batch_first=True)
        elif self.encoder == 'gru':
            self.rnn = nn.GRU(self.emsize, self.nhid, self.nlayers, batch_first=True)
        else:
            raise KeyError('Unsupported encoder!!!')

        # attention unit: from hidden unit to task-specific attention weight
        self.attn_unit = nn.Linear(self.nhid, 3, bias=False)
        torch.nn.init.xavier_uniform_(self.attn_unit.weight)

        # output for specific task
        self.h2o_time = nn.Linear(self.nhid, 1, bias=False)
        self.h2o_action = nn.Linear(self.nhid, self.num_type, bias=False) 
        self.h2o_item = nn.Linear(self.nhid, self.num_topic, bias=False)

        self.h2o_time_bias = torch.nn.Parameter(torch.rand(self.num_seq, 1))
        self.h2o_action_bias = torch.nn.Parameter(torch.rand(self.num_seq, self.num_type))
        self.h2o_item_bias = torch.nn.Parameter(torch.rand(self.num_seq, self.num_topic))

        torch.nn.init.xavier_uniform_(self.h2o_time.weight)
        torch.nn.init.xavier_uniform_(self.h2o_action.weight)
        torch.nn.init.xavier_uniform_(self.h2o_item.weight)

        # time-decaying parameter
        self.w = torch.nn.Parameter(torch.rand(self.num_seq, 3) - 0.5)
        self.alpha = torch.nn.Parameter(torch.rand(3))

        if self.self_embedding:
            self.vocab_emb = torch.nn.Parameter(torch.rand((self.num_prod, self.item_emsize)) - 0.5)
        else:
            self.vocab_emb = config['vocab_emb']

        self.topic_emb = torch.nn.Parameter(torch.rand((self.num_topic, self.item_emsize)) - 0.5)
        
        self.sigm = nn.Sigmoid()
        self.sftp = nn.Softplus()
        self.drop = nn.Dropout(self.dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def fourier_feature_transform(self, time_intervals):
        """
        Map scalar intervals or action indices to Fourier features.

        The log1p transform stabilizes large values, and sine/cosine bases
        provide a smooth high-dimensional representation for continuous-time
        sequence modeling.
        """
        if self.item_emsize % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {self.item_emsize}")

        # Stabilize the input scale before applying periodic bases.
        log_time = torch.log1p(time_intervals)

        # Build a bank of frequencies from low to high resolution.
        half_dim = self.item_emsize // 2
        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float) * -(math.log(100.0) / half_dim))
        div_term = div_term.to(log_time.device)
        
        # Broadcast each scalar value against the frequency bank.
        args = log_time.unsqueeze(-1) * div_term

        sin_features = torch.sin(args)
        cos_features = torch.cos(args)

        return torch.cat([sin_features, cos_features], dim=-1)

    def forward(self, idx, data, hidden):
        """
        Compute SJ-NPP outputs for one user's event history.

        Args:
            idx: user's idx
            data (dict): a dictionary containing a user's event data
            hidden (pytorch tensor): initial values of hidden states
            vocab_emb (pytorch tensor): embedding of all items
            
        Returns:
            out_time
            out_action
            out_item
            out
            ejc_decay
            aw
        """
        item_emb = self.vocab_emb[data['item'].int()]
        len_seq = len(data['time'])

        if self.action_encoding == 'one-hot':
            action = F.one_hot(data['action'].long(), num_classes=self.num_type)
        else:
            action = data['action']

        delta_t = data['delta_t']
        elapsed_t = data['elapsed_t']

        # Encode when/how/what event information into a shared feature space.
        time_emb = self.fourier_feature_transform(delta_t)
        action_emb = self.fourier_feature_transform(action)
        feature_stacked = torch.stack([time_emb, action_emb, item_emb], dim=1)

        # Let time, action, and item features attend to one another before
        # fusing them into the recurrent event representation.
        attn_output, _ = self.inter_feature_attention(query=feature_stacked, key=feature_stacked, value=feature_stacked)
        contextualized_features_flat = attn_output.reshape(len_seq, -1)
        x = self.fusion_mlp_after_attn(contextualized_features_flat)

        # Recurrent encoder summarizes the observed e-commerce event sequence.
        out = self.rnn(self.drop(x).unsqueeze(dim=0), hidden)[0].squeeze(dim=0)

        # Task-specific temporal attention: previous hidden states receive
        # different decayed weights for time, action, and item prediction.
        mask = (torch.arange(len_seq).unsqueeze(1) - torch.arange(len_seq).unsqueeze(0) > 0).to(self.device)
        time_decay = torch.exp(-self.sftp(self.w[idx]).reshape(self.w[idx].shape[0], 1, 1) * elapsed_t.unsqueeze(dim=0))
        time_decay = time_decay * mask.float()

        ejc_decay = torch.exp(self.sftp(self.attn_unit(out)).transpose(1, 0).unsqueeze(dim=1) * time_decay)
        ejc_decay = ejc_decay * mask.float()

        sum_ejc = ejc_decay.sum(dim=-1, keepdim=True)
        sum_ejc = torch.where(sum_ejc==0, 1.0, sum_ejc)
        
        aw = ejc_decay / sum_ejc
        sn = aw.matmul(out)

        zero_column = torch.zeros_like(out[:1, :])
        out = torch.cat([zero_column, out], dim=0)

        # Joint outputs: event intensity for when, action distribution for how,
        # and item distribution for what.
        out_time = self.sftp(self.h2o_time(sn[0] + out[:-1]) + self.h2o_time_bias[idx])
        out_action_pre = self.h2o_action(sn[1] + out[:-1])  + self.h2o_action_bias[idx]
        out_item_pre = self.h2o_item(sn[2] + out[:-1]) + self.h2o_item_bias[idx]

        out_action = F.softmax(out_action_pre, dim=1)

        out_item = F.softmax(F.softmax(out_item_pre, dim=1) @ self.topic_emb @ self.vocab_emb.T, dim=1) 

        return out_time, out_action, out_item, out, ejc_decay, aw