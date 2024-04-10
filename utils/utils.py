import os
import torch
from model import ARPESNet
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms import Normalize
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

def get_partial_sample(X, y, ratio, stratify=True):
    """
    Returns a partial sample of the dataset.
    """
    if stratify:
        idx, _ = train_test_split(np.arange(len(y)), test_size=1-ratio, random_state=42, stratify=y)
    else:
        idx = np.random.choice(len(y), int(len(y)*ratio), replace=False)
    return X[idx], y[idx]