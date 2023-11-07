import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, Dropout, BatchNorm2d, BatchNorm1d

class ARPESNet(nn.Module): 
    def __init__(self, 
                 num_classes: int = 3,
                 in_channels: int = 1,
                 hidden_channels=32,
                 dropout=0.0, 
                 fcw=50,  
                 kernel_size=3,
                ): 
        super().__init__()
        self.convs = Sequential(
            Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            #BatchNorm2d(hidden_channels),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            #BatchNorm2d(hidden_channels),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            #BatchNorm2d(hidden_channels),
            #Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            #ReLU(),
            #MaxPool2d(kernel_size=kernel_size, stride=2),
            #Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            #ReLU(),
            #MaxPool2d(kernel_size=kernel_size, stride=2),
            #BatchNorm2d(hidden_channels),
            )
        self.flat = nn.Flatten()
        self.dropout = Dropout(dropout)

        x = torch.empty(1, 1, 400, 195)
        x = self.convs(x)
        x = self.flat(x)
        out_dim = x.size(-1)

        self.fc1 = Linear(out_dim, out_features=fcw)
        self.fc2 = Linear(in_features=fcw, out_features=fcw)
        self.fc3 = Linear(in_features=fcw, out_features=num_classes)

    def forward(self, x):
        x = self.convs(x)
        x = self.flat(x)
        x = self.fc1(self.dropout(x))
        #x = self.batch_norm1(x)
        x = ReLU()(x)
        x = self.fc2(self.dropout(x))
        #x = self.batch_norm2(x)
        x = ReLU()(x)
        x = self.fc3(self.dropout(x))

        return x