from typing import Callable, Optional
import numpy as np
from torchvision.datasets import VisionDataset


class ARPESDataset(VisionDataset):
    def __init__(
        self, 
        data,
        targets: Optional[np.ndarray] = None,
        normalize: bool = False,
        transform: Optional[Callable] = None,
        ):
        super().__init__(root=None, transform=transform)
        self.data, self.targets = data, targets
        self.if_normalize = normalize
        
    def __getitem__(self, index):
        X = self.data[index]
        if self.transform is not None:
            X = self.transform(X).squeeze(0)        
        if self.targets is not None:
            y = self.targets[index]
            if self.target_transform is not None:
                y = self.target_transform(self.targets[index])
            return X, y, index
        else:
            return X, index
    
    def __len__(self):
        return len(self.data)