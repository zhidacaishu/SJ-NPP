import torch
from torch import trapz
import torch.nn.functional as F

def predict(model, data, idx, up_lim):
    time = data['time'][idx]
    action = data['action'][idx]
    item = data['item'][idx]
    delta_t = data['delta_t'][idx]
    elapsed_t = data['elapsed_t'][idx]
    end_idx = data['train_split_idx'][idx]

    model.eval()
    with torch.no_grad():
        input_time = time[:end_idx]
        input_action = action[:end_idx]
        input_item = item[:end_idx]
        input_delta_t = delta_t[:end_idx]
        input_elapsed_t = elapsed_t[:end_idx]

        device = input_time.device
        if model.encoder == 'lstm':
            hidden = (
                torch.zeros(model.nlayers, 1, model.nhid, device=device),
                torch.zeros(model.nlayers, 1, model.nhid, device=device)
            )
        elif model.encoder == 'gru':
            hidden = torch.zeros(model.nlayers, 1, model.nhid, device=device)

        input_data = {
            'time': input_time,
            'action': input_action,
            'item': input_item,
            'delta_t': input_delta_t,
            'elapsed_t': input_elapsed_t[:input_elapsed_t.shape[0], :input_elapsed_t.shape[0]]
        }

        _, _, _, out, _, aw = model(idx, input_data, hidden)
        _ = aw

        next_time = pred_time(input_time, up_lim, out, idx, model)
        next_action, next_action_prob = pred_action(next_time, input_time, out, idx, model)
        user_topic_dist, next_item, next_item_prob = pred_item(next_time, input_time, out, idx, model)

    return next_time, next_action, next_action_prob, user_topic_dist, next_item, next_item_prob

def pred_time(input_time, up_lim, out, idx, model, n_points=20000):
    model.eval()
    with torch.no_grad():
        device = input_time.device
        t_start = input_time[-1]
        t_end = t_start + up_lim
        t_points = torch.linspace(t_start, t_end, n_points, device=device)

        # Evaluate the time-decayed attention history over integration points.
        elapsed_times = t_points.unsqueeze(1) - input_time.unsqueeze(0)  # (n_points, seq_len)
        mask = elapsed_times > 0

        elapsed_times = elapsed_times * mask

        time_decays = torch.exp(-model.sftp(model.w[idx, 0]) * elapsed_times) * mask  # (n_points, seq_len)

        # Compute task-specific attention weights for the time prediction head.
        ejc_decays = out[1:].matmul(model.attn_unit.weight[0])  # (seq_len,)

        ejc_decays = torch.exp(time_decays * model.sftp(ejc_decays).unsqueeze(0)) * mask  # (n_points, seq_len)
        sum_ejcs = ejc_decays.sum(dim=1, keepdim=True)  # (n_points, 1)
        sum_ejcs = torch.where(sum_ejcs == 0, 1.0, sum_ejcs)
        aws = ejc_decays / sum_ejcs  # (n_points, seq_len)

        # Aggregate hidden states and evaluate the conditional intensity.
        sns = aws.matmul(out[1:])  # (n_points, hidden_dim)

        lambda_vals = model.sftp(model.h2o_time(sns + out[-1]) + model.h2o_time_bias[idx]).squeeze(1)  # (n_points,)

        # Numerically integrate the cumulative intensity.
        x_diffs = t_points.diff(dim=-1)
        y_vals = (lambda_vals[:-1] + lambda_vals[1:]) / 2
        areas = x_diffs * y_vals
        integral_lambda = F.pad(torch.cumsum(areas, dim=0), (1, 0))

        # Expected event time under the conditional density f*(t).
        fstar_vals = t_points * lambda_vals * torch.exp(-integral_lambda)

        pred_time_integral = trapz(fstar_vals, t_points)

        return pred_time_integral.item()

def pred_action(t, input_time, out, idx, model):
    model.eval()
    with torch.no_grad():
        t = torch.tensor([t], device=input_time.device)

        elapsed_t = t - input_time
        pos_ind = elapsed_t > 0

        elapsed_t = elapsed_t * pos_ind
        time_decay = torch.exp(-model.sftp(model.w[idx, 1])*elapsed_t) * pos_ind

        ejc_decay = out[1:].matmul(model.attn_unit.weight[1])
        ejc_decay = torch.exp(time_decay * model.sftp(ejc_decay)) * pos_ind

        sum_ejc = ejc_decay.sum()
        sum_ejc = (sum_ejc == 0).float() * 1 + (sum_ejc != 0).float() * sum_ejc
        aw = ejc_decay / sum_ejc
        sn = aw.matmul(out[1:])
        
        # Convert the action head output into a categorical distribution.
        out_action_pre = model.h2o_action(sn + out[-1]) + model.h2o_action_bias[idx]
        out_action = F.softmax(out_action_pre, dim=0)
        if not torch.isfinite(out_action).all():
            raise RuntimeError(f'Non-finite action probabilities for sequence {idx}.')

        pred_action = torch.multinomial(out_action, num_samples=1, replacement=True) 

    return pred_action, out_action


def pred_item(t, input_time, out, idx, model):
    model.eval()
    with torch.no_grad():
        t = torch.tensor([t], device=input_time.device)

        elapsed_t = t - input_time
        pos_ind = elapsed_t > 0

        elapsed_t = elapsed_t * pos_ind
        time_decay = torch.exp(-model.sftp(model.w[idx, 2])*elapsed_t) * pos_ind

        ejc_decay = out[1:].matmul(model.attn_unit.weight[2])
        ejc_decay = torch.exp(time_decay * model.sftp(ejc_decay)) * pos_ind

        sum_ejc = ejc_decay.sum()
        sum_ejc = (sum_ejc == 0).float() * 1 + (sum_ejc != 0).float() * sum_ejc
        aw = ejc_decay / sum_ejc
        sn = aw.matmul(out[1:])

        out_item_pre = model.h2o_item(sn + out[-1]) + model.h2o_item_bias[idx]
        user_topic_dist = F.softmax(out_item_pre, dim=0) 

        out_item = F.softmax(user_topic_dist @ model.topic_emb @ model.vocab_emb.T, dim=0)
  
        pred_item = torch.multinomial(out_item, num_samples=1, replacement=True)

    return user_topic_dist, pred_item, out_item