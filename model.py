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
                 dropout=0., 
                 fcw=50,  
                 kernel_size=3,
                ): 
        super().__init__()
        '''       
        self.conv1 = Sequential(Conv2d(in_channels, hidden_channels, 1), 
                                #MaxPool2d(kernel_size=kernel_size, stride=1), 
                                ReLU())
        self.conv2 = Sequential(Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1), 
                                #MaxPool2d(kernel_size=kernel_size, stride=2), 
                                ReLU())
        self.conv3 = Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2)
        self.conv4 = Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2)
        '''
        self.convs = Sequential(
            Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),         
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),        
            Conv2d(in_channels=hidden_channels*2, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels//2, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),
            #Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            #ReLU(),
            #MaxPool2d(kernel_size=kernel_size, stride=2),
            #BatchNorm2d(hidden_channels),
            )
 
        self.flat = nn.Flatten()
        self.dropout = Dropout(dropout)

        x = torch.empty(1, 1, 400, 195)
        out = self.convs(x)
    
        h_shape = out.size()[1:]
        #h_shape: torch.Size([32, 50, 25])
        out = self.flat(out)
        out_dim = out.size(-1)

        self.domain_classifier = Sequential(
            Linear(out_dim, fcw),
            ReLU(),
            self.dropout,
            nn.Linear(fcw, fcw),
            ReLU(),
            self.dropout,
            nn.Linear(fcw, 2),
        )
        self.class_classifier = Sequential(
            Linear(out_dim, fcw),
            ReLU(),
            self.dropout,
            nn.Linear(fcw, fcw),
            ReLU(),
            self.dropout,
            nn.Linear(fcw, num_classes),

        )

    def forward(self, x, alpha):
        out = self.convs(x)
        out = self.flat(out)
        out = self.dropout(out)
        reverse_x = ReverseLayerF.apply(out, alpha)
        domain_out = self.domain_classifier(reverse_x)
        class_out = self.class_classifier(out)

        return class_out, domain_out