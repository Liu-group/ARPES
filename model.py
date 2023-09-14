import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, Dropout

class ARPESNet(nn.Module): 
    def __init__(self, 
                 num_classes: int = 3,
                 in_channels: int = 1,
                 hidden_channels=32,
                 dropout=0.0, 
                 fcw=50,  
                 kernel_size=3,
                 predict=False
                ): 
        super().__init__()
        self.predict = predict
        self.convs = Sequential(
            Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=2)
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
        print(x)
        if self.predict:
            # load the mean and std 
            train_mean, train_std = torch.load('mean_std.pt')
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            std_nonzero = std.nonzero(as_tuple=True)[0]
            x[:, std_nonzero] = (x[:, std_nonzero] - mean[std_nonzero]) / std[std_nonzero]
            x = x * train_std + train_mean
        print(x.shape)
        print(x)

        x = self.fc1(self.dropout(x))
        x = ReLU()(x)
        x = self.fc2(self.dropout(x))
        x = ReLU()(x)
        x = self.fc3(self.dropout(x))

        return x