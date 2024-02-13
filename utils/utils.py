import os
import torch
from model import ARPESNet
import numpy as np
import random

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
    if args.checkpoint_path == './checkpoint/':
        save_path = os.path.join(args.checkpoint_path, str(args.num_classes) + '_' + str(args.seed) + f'_model_{args.adaptation}.pt')
    else:
        save_path = os.path.join(args.checkpoint_path, str(args.num_classes) + '_' + str(args.seed) + '_model.pt')
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