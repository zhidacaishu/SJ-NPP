import time as tm
import torch
from utils import EarlyStopping, timeSince, Struct
from loss import loss_action, loss_item, loss_time
import torch.optim as optim

def train_model(data, model, args):
    """
    function for training the main model

    Args:
        data (dict): a dictionary of pytorch tensors for input data
            time: time of events
            action: action type of user
            item: subject item of action
            delta_t: time difference of consecutive events
            elapsed_t: time difference between any two events
            eff_seqlen: length of user event sequences 
        model (pytorch model): pytorch model object of an untrained model
        args (dict): a python dictionary storing all other args used in the function;
            see get_args function listed in utils.py for all the used arguments

    Returns:
        model (pytorch model): pytorch model object of a trained model
        new_result_dict (dict): a dictionary of training logs generated from the 
            training procedure. It contains a model's performance regarding 
            loss, rmse, mae, precision, recall, f1, and roc auc on training, 
            validation, and testing data.
    """
    if type(args) == dict:
        args = Struct(**args)
    
    device = torch.device(args.device)

    time = data['time']
    item = data['item']
    action = data['action']
    delta_t = data['delta_t']
    elapsed_t = data['elapsed_t']
    
    assert (args.loss == 'time' or args.loss == 'time+action' or args.loss == 'time+item' or  args.loss == 'all')

    result_dict = {}
    result_dict['loss'] = []

    early_stopping = EarlyStopping({
        'patience': args.patience,
        'verbose' : True,
        'ckpt_dir': args.ckpt_dir,
        'ckpt_fn' : args.ckpt_fn,
        'result_dir': args.result_dir,
        'result_fn' : args.result_fn
    })

    print("Finish creating early stopping object! Criterion based on " + args.criterion + " dataset.")

    start = tm.time()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.penalty)

    for epoch in range(args.epochs):
        model.train()
        idx_shuffled = torch.randperm(data['num_seq'])

        tmp_eval = 0.0
        for i in range(data['num_seq']):
            idx = idx_shuffled[i]
            train_cutoff = data['train_split_idx'][idx]
            
            if train_cutoff == 0:
                continue

            train_time = time[idx, :train_cutoff]
            train_action = action[idx, :train_cutoff]
            train_item = item[idx, :train_cutoff]
            train_delta_t = delta_t[idx, :train_cutoff]
            train_elapsed_t = elapsed_t[idx, :train_cutoff]

            if args.encoder == 'lstm':
                hidden = (torch.zeros(args.nlayers, 1, args.nhid, device=device), torch.zeros(args.nlayers, 1, args.nhid, device=device))
            elif args.encoder == 'gru':
                hidden = torch.zeros(args.nlayers, 1, args.nhid, device=device)

            input_data = {
                'time': train_time,
                'action': train_action,
                'item': train_item,
                'delta_t': train_delta_t,
                'elapsed_t': train_elapsed_t[:train_elapsed_t.shape[0], :train_elapsed_t.shape[0]]
            }

            out_time, out_action, out_item, out, _, aw = model(idx, input_data, hidden)                  
            _ = aw

            if args.loss == 'time':
                tmp_loss_ts  = loss_time(idx, train_time, out_time, out, model, sample=50)
                tmp_loss_action = 0.0
                tmp_loss_item = 0.0
            elif args.loss == 'time+action':
                tmp_loss_ts  = loss_time(idx, train_time, out_time, out, model, sample=50)
                tmp_loss_action = loss_action(train_action, out_action)
                tmp_loss_item = 0.0
            elif args.loss == 'time+item':
                tmp_loss_ts  = loss_time(idx, train_time, out_time, out, model, sample=50)
                tmp_loss_action = 0
                tmp_loss_item = loss_item(train_item, out_item)
            elif args.loss == 'all':
                tmp_loss_ts  = loss_time(idx, train_time, out_time, out, model, sample=50)
                tmp_loss_action = loss_action(train_action, out_action)
                tmp_loss_item = loss_item(train_item, out_item)
            else:
                raise KeyError('unsupported loss!')
            
            tmp_loss = tmp_loss_ts + tmp_loss_action + tmp_loss_item 
            tmp_eval += tmp_loss.item()

            model.zero_grad()
            tmp_loss.backward()
            optimizer.step()

        tmp_eval /= data['num_seq']
        result_dict['loss'].append(tmp_eval)
        
        print('({:s}) Finished epoch {} training!'.format(timeSince(start), epoch+1))

        early_stopping(tmp_eval, model, result_dict)

        if early_stopping.early_stop or torch.isnan(tmp_loss) or torch.isinf(tmp_loss):
            break
    
    if early_stopping.early_stop:
        exit_status = 0
        print("Early stopping! Total time used: {:s}, golden epoch: {}".format(timeSince(start), epoch - early_stopping.counter+1))
    elif torch.isnan(tmp_loss):
        exit_status = 1
        print("NaN loss value encountered! Total time used: {:s}, golden epoch: {}".format(timeSince(start), epoch - early_stopping.counter+1))
    elif torch.isinf(tmp_loss):
        exit_status = 2
        print("Inf loss value encountered! Total time used: {:s}, golden epoch: {}".format(timeSince(start), epoch - early_stopping.counter+1))        
    else:
        exit_status = 3
        print("Reach maximum epoch number! Total time used: {:s}, golden epoch: {}".format(timeSince(start), epoch - early_stopping.counter+1))

    new_result_dict = {}
    new_result_dict['exit_status'] = exit_status

    return model, new_result_dict