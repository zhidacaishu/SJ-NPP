import pandas as pd
import torch
from tqdm import tqdm

def data_load(args, dev=True):
    '''
    Function used to load and preprocess the data

    Args:
        args (dict): main arguments used for data loading & processing
            data_file (str): path to the data file
            train_split (float): percentage of data used for training
            valid_split (float): percentage of data used for validation
            
        dev (bool): whether using a sample of data for fast development
        
    Returns:
        data (dict): a dictionary contains preprocessed data and its metadata
            time (pytorch tensor): elapsed time of the event since 2014-04-01 00:00:00
            action (pytorch tensor): users' action type
            item (pytorch tensor): processed item index 
            delta_t (pytorch tensor): time difference of consecutive events
            elapsed_t (pytorch tensor): time difference between any two events
    '''

    if type(args) != dict:
        args = vars(args)

    # unpack arguments from args
    data_file = args['data_file']
    train_split = args['train_split']
    valid_split = args['valid_split']
    unit = args['unit']
    device = args.get('device', 'cpu')

    # value used for padding unobserved events
    padding_idx = 9999999

    data = csv_to_torch(data_file, device=device)

    # Use a subset of user sequences for fast development when requested.
    if dev == True:
        sample_idx = torch.arange(data.shape[0])[:5000]
    else:
        sample_idx = torch.arange(data.shape[0])

    time = data[sample_idx, :, 0]
    action = data[sample_idx, :, 1]
    item = data[sample_idx, :, 2]

    mask = (action == padding_idx)

    # preprocess data
    # time: measure with seconds
    time = time_unit(time, unit)
    time[mask] = padding_idx

    # compute time differences between consequent events
    delta_t = torch.cat((time[:, 0].unsqueeze(dim=1), time[:, 1:] - time[:, :-1]), dim=-1)
    delta_t[mask] = padding_idx

    # compute the elapsed time since the activation
    elapsed_t = time.unsqueeze(dim=-1) - time.unsqueeze(dim=1)
    elapsed_t[elapsed_t < 0] = 0
    elapsed_t[mask] = padding_idx

    # elapsed_t: shape U*T*T
    # u = 1:
    # [[0      , 0      , 0, ...],
    #  [t1 - t0, 0      , 0, ...],
    #  [t2 - t0, t2 - t1, 0, ...]
    #  ...]

    data = {
        'time': time,
        'action': action,
        'item': item,
        'delta_t': delta_t,
        'elapsed_t': elapsed_t
    }

    # number of sequences
    data['num_seq'] = data['action'].shape[0]
    # number of event types
    num_type = torch.unique(data['action']).shape[0] - 1
    data['num_type'] = torch.tensor(num_type)

    eff_seqlen, train_split_idx, valid_split_idx = data_split(action, train_split, valid_split)
    data['eff_seqlen'] = eff_seqlen
    data['train_split_idx'] = train_split_idx
    data['valid_split_idx'] = valid_split_idx

    return data

def csv_to_torch(data_file, device='cpu'):
    '''
    Function used to convert csv data of user activities to torch tensors
    
    Args:
        data_file: path to the data file
        
    Returns:
        torch_data (pytorch tensor): converted tensor of users' event sequences
    '''
    # Read event sequence data from CSV.
    csv_data = pd.read_csv(data_file)

    # Identify users and allocate a padded tensor for all sequences.
    unique_users = csv_data['User_id'].unique()
    num_seq = len(unique_users)

    max_seq_len = csv_data.groupby('User_id').size().max()

    torch_data = torch.full((num_seq, max_seq_len, 3), 9999999.)

    grouped = csv_data.groupby('User_id')

    # Fill each user's event sequence with time, action, and item id.
    for user_id in tqdm(unique_users):
        user_data = grouped.get_group(user_id)
        num_events = user_data.shape[0]

        time = user_data['Time'].to_numpy()
        action = user_data['Action'].to_numpy()
        item_id = user_data['Item_id'].to_numpy()

        torch_data[user_id, :num_events, 0] = torch.tensor(time, dtype=torch.float32)
        torch_data[user_id, :num_events, 1] = torch.tensor(action, dtype=torch.float32)
        torch_data[user_id, :num_events, 2] = torch.tensor(item_id, dtype=torch.float32)

    torch_data = torch_data.to(device)

    return torch_data
    
def data_split(action, train_split, valid_split):
    """
    splits the preprocessed data to training, validation, and test sets
    Args:
        action (pytorch tensor): type of action
        train_split (float): percentage of data used for training
        valid_split (float): percentage of data used for validation
       
    Returns:
        eff_seqlen (pytorch tensor): sequence length for all users
        train_split_idx (pytorch tensor): training data indicators for all users
        valid_split_idx (pytorch tensor): validation data indicators for all users
    """
    padding_idx = 9999999

    # number of sequences/users
    num_seq = action.shape[0]
    # maximum sequence length
    len_seq = action.shape[1]

    # get effective sequence lengths (events that are not padded)
    mask = (action == padding_idx)
    eff_seqlen = (len_seq - mask.sum(dim=1)).int()

    # get number of user activities
    act_idx = (~mask).float()

    # Split each sequence into training and validation/test segments.
    train_split_idx = torch.argmax((torch.cumsum(act_idx, dim=1) == (torch.ceil(eff_seqlen * train_split)).unsqueeze(dim=-1)).int(), dim=-1).int()
    valid_split_idx = torch.argmax((torch.cumsum(act_idx, dim=1) == (torch.ceil(eff_seqlen * (train_split + valid_split))).unsqueeze(dim=-1)).int(), dim=-1).int() + 1

    return eff_seqlen, train_split_idx, valid_split_idx

def time_std(time, proc_type):
    pass 

def time_unit(time, unit):
    """transforms tht unit of time
    Args: 
        time (pytorch tensor): raw data of time
        unit (str): how to transform data, either day, hour, minute or no transformation (second) 

    Returns:
        time(pytorch tensor): transformed time
    """
    padding_idx = 9999999

    mask = (time == padding_idx)

    if unit == 'no':
        time = time
    elif unit == 'day':
        time[~mask] = time[~mask] / 86400 
    elif unit == 'hour':
        time[~mask] = time[~mask] / 3600
    elif unit == 'min':
        time[~mask] = time[~mask] / 60
    else:
        raise ValueError('Time unit not supported!')
    
    return time