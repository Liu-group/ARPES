import os
import torch
from model import ARPESNet
import inspect
from argparse import Namespace

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

def load_checkpoint(args):
    """
    Loads a model checkpoint.
    """
    assert os.path.exists(args.checkpoint_path), "Checkpoint not found"
    if args.gpu is not None:
        state = torch.load(args.checkpoint_path)
    else:
        state = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model = state["full_model"]

    return model


    