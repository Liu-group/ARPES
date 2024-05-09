import os
import torch
from model import ARPESNet
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    """Sets initial seed for random numbers."""
    # set random seed for torch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_checkpoint(args):
    """
    Loads a model checkpoint.
    """
    save_path = os.path.join(args.checkpoint_path, str(args.num_classes) + '_' + str(args.seed) + f'_model_{args.adaptation}.pt')
    assert os.path.exists(save_path), f"Checkpoint {save_path} not found"
    print('Loading checkpoint from: ' + save_path)
    if args.gpu is not None:
        state = torch.load(save_path )
    else:
        state = torch.load(save_path , map_location=torch.device('cpu'))
    if args.mode == 'predict':
        model = ARPESNet(num_classes=args.num_classes,
                         hidden_channels=args.hidden_channels, 
                         conditional=args.conditional,
                         prediction_mode=True,
                        )
        model.load_state_dict(state["state_dict"])
    else:
        model = state["full_model"]

    return model

def normalize_transform(name):
    """
    Normalizes the input tensor.
    """
    if name == 'sim':
        transform = transforms.Compose([transforms.ToTensor(),
                                        Normalize((1.000,), (2.517)),
                                        ])
    elif name == 'exp_2014':
        transform = transforms.Compose([transforms.ToTensor(),
                                        Normalize((1.000,), (1.754))
                                        ])
    elif name == 'exp_2015':
        transform = transforms.Compose([transforms.ToTensor(),
                                        Normalize((1.000,), (1.637))
                                        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        Normalize((1.000,), (1.699))
                                        ])
    return transform

def get_partial_sample(X, y, ratio, stratify=False):
    """
    Returns a partial sample of the dataset.
    """
    if stratify:
        idx, _ = train_test_split(np.arange(len(y)), test_size=1-ratio, random_state=42, stratify=y)
    else:
        idx = np.random.choice(len(y), int(len(y)*ratio), replace=False)
    return X[idx], y[idx]

def get_num_sample(X, y, num, stratify=False):
    """
    Returns a partial sample of the dataset.
    """
    if stratify:
        idx, _ = train_test_split(np.arange(len(y)), test_size=num/len(y), random_state=42, stratify=y)
    else:
        idx = np.random.choice(len(y), num, replace=False)
    print(f"Number of target samples used: {len(idx)}")
    # get number of samples for each value of y
    unique, counts = np.unique(y[idx], return_counts=True)
    print(dict(zip(unique, counts)))

    return X[idx], y[idx]

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def next(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)