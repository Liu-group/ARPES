import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, Dropout, BatchNorm2d, BatchNorm1d
from functions import ReverseLayerF
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
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),         
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),        
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),
            #Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            #MaxPool2d(kernel_size=kernel_size, stride=2),
            #ReLU(),
            #Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            #ReLU(),
            #MaxPool2d(kernel_size=kernel_size, stride=2),
            #BatchNorm2d(hidden_channels),
            )
        self.flat = nn.Flatten()
        self.dropout = Dropout(dropout)

        x = torch.empty(1, 1, 400, 195)
        x = self.convs(x)
        h_shape = x.size()[1:]
        #h_shape: torch.Size([32, 50, 25])
        x = self.flat(x)
        out_dim = x.size(-1)

        self.domain_classifier = Sequential(
            Linear(out_dim, fcw),
            ReLU(),
            nn.Linear(fcw, fcw),
            ReLU(),
            nn.Linear(fcw, num_classes),
        )
        self.class_classifier = Sequential(
            Linear(out_dim, fcw),
            ReLU(),
            nn.Linear(fcw, fcw),
            ReLU(),
            nn.Linear(fcw, num_classes),

        )

    def forward(self, x, alpha):
        x = self.convs(x)
        x = self.flat(x)
        reverse_x = ReverseLayerF.apply(x, alpha)
        domain_out = self.domain_classifier(reverse_x)
        class_out = self.class_classifier(x)

        return class_out, domain_out