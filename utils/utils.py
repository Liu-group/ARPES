import os
import torch
from model import ARPESNet
import inspect
from argparse import Namespace
from typing import Any, Dict, List, Tuple
import torch.nn as nn
from torch import Tensor
import numpy as np

from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

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
    save_path = os.path.join(args.checkpoint_path, str(args.seed) + '_model.pt')
    assert os.path.exists(save_path), "Checkpoint not found"
    print('Loading checkpoint from: ' + save_path)
    if args.gpu is not None:
        state = torch.load(save_path )
    else:
        state = torch.load(save_path , map_location=torch.device('cpu'))
    if args.mode == 'predict':
        model = ARPESNet(num_classes=args.num_classes)
        model.load_state_dict(state["state_dict"])
    else:
        model = state["full_model"]

    return model

def kl_divergence(target, inputs):
    return torch.sum(torch.exp(target) * (target - inputs))


    