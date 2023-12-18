from typing import Callable, Optional
import numpy as np
from torchvision.datasets import VisionDataset


class ARPESDataset(VisionDataset):
    def __init__(
        self, 
        data,
        targets,
        normalize: bool = False,
        transform: Optional[Callable] = None,
        ):
        super().__init__(root=None, transform=transform)
        self.data, self.targets = data, targets
        self.if_normalize = normalize
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.max = np.max(self.data, axis=0)
        self.min = np.min(self.data, axis=0)
    
    def normalize(self, data):
        return (data-self.mean)/self.std
    def standardize(self, data):
        return (data-self.min)/(self.max-self.min)
        
    def __getitem__(self, index):
        X, y = self.data[index], self.targets[index]
        if self.transform is not None:
            X = self.transform(X).squeeze(0)
        if self.target_transform is not None:
            y = self.target_transform(self.targets[index])
        if self.if_normalize:
            return self.normalize(X), y, index
        else:
            return X, y, index
    
    def __len__(self):
        return len(self.data)