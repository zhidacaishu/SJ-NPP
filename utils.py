import os
import torch
import numpy as np
import random
import argparse
import math
import time
import pickle


def get_args():
    """
    Argument parser for running main.py in command line
    """
    parser = argparse.ArgumentParser()

    # parser for model hyperparameters
    parser.add_argument('--encoder', type=str, default='gru', 
                        help='which encoder to use, gru or lstm?')
    parser.add_argument('--emsize', type=int, default=128,
                        help='size of feature embeddings')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers for the RNN')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout rate')

    # parser for data loader
    parser.add_argument('--data_file', type=str, default='./data/your_processed_sequences.csv',
                        help='path to the data file')
    parser.add_argument('--proc_type', type=str, default='z-score',
                        help='how to pre-process the data, e.g., z-score, min-max, or no?')
    parser.add_argument('--train_split', type=float, default=0.6,
                        help='percentage of events used for model training')
    parser.add_argument('--valid_split', type=float, default=0.2,
                        help='percentage of events used for model validation')
    # parser for optimizer
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--penalty', type=float, default=.0001, 
                        help='penalty for optimizer')
    parser.add_argument('--epochs', type=int, default=10,
                        help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stopping')
    parser.add_argument('--criterion', type=str, default='valid',
                        help='criterion for early stopping, valid, train, or test?')
    # parser for general information
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='which device to use, cpu or cuda?')
    parser.add_argument('--result_dir', type=str, default='./result/',
                        help='where to save training results/log?')
    parser.add_argument('--ckpt_dir', type=str, default='./models/',
                        help='where to save the trained model?')

    return parser.parse_args()


class Struct:
    """
    Function to convert a python dictionary to a python struct
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

        
def create_dir(dirname):
    """
    Function to create a directory if it does not already exist.
    
    Args:
        dirname: directory to create
    
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def setup_seed(seed):
    """
    Function to set up the random seed for reproducibility
    
    Args:
        seed (int): seed used for the code
    
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def timeSince(since):
    """
    Function to show the elapsed time in minutes
    
    Args:
        since (float): starting time for comparison
    
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class EarlyStopping:
    """
    Early stops the training if validation loss 
    doesn't improve after a given patience.
    """
    def __init__(self, config):
        """
        Args:
            ckpt_dir: the directory where you want to store the model checkpoint.
            ckpt_fn:  the file name for the saved the model checkpoint.
            result_dir: the directory where you want to store the results.
            result_fn:  the file name for the result.
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """

        self.patience = config['patience']
        self.verbose = config['verbose']
        self.ckpt_dir = config['ckpt_dir']
        self.ckpt_fn  = config['ckpt_fn']
        self.result_dir = config['result_dir']
        self.result_fn  = config['result_fn']

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, result_dict):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save(val_loss, model, result_dict)
        elif score < self.best_score * 1.005:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save(val_loss, model, result_dict)
            self.counter = 0

    def save(self, val_loss, model, result_dict):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # create directory if not exists
        create_dir(self.ckpt_dir)
        create_dir(self.result_dir)

        # save estimated model
        torch.save(model, self.ckpt_dir + self.ckpt_fn)

        # save training result/log
        with open(self.result_dir + self.result_fn, 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.val_loss_min = val_loss