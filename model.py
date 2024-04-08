import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, Dropout, LeakyReLU, PReLU
from functions import ReverseLayerF

class ARPESNet(nn.Module): 
    def __init__(self, 
                 num_classes: int = 3,
                 in_channels: int = 1,
                 hidden_channels=32,
                 negative_slope=0.01,
                 dropout=0., 
                 fcw=50,  
                 kernel_size=3,
                 prediction_mode = False,
                ): 
        super().__init__()
        self.convs = Sequential(
            Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            LeakyReLU(negative_slope=negative_slope),   
            #PReLU(),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),        
            Conv2d(in_channels=hidden_channels*2, out_channels=hidden_channels, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),
            Conv2d(in_channels=hidden_channels, out_channels=hidden_channels//2, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),
            )

        self.prediction_mode = prediction_mode
        self.dropout = Dropout(dropout)
        x = torch.empty(1, 1, 400, 195)
        out = self.convs(x)
    
        h_shape = out.size()[1:]
        #h_shape: torch.Size([32, 50, 25])
        out = out.view(1, -1)
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

    def forward(self, x, alpha=1.):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        reverse_x = ReverseLayerF.apply(out, alpha)
        domain_out = self.domain_classifier(reverse_x)
        class_out = self.class_classifier(out)

        if self.prediction_mode:
            return class_out
        else:
            return class_out, domain_out

class ModelSoftmaxWrapper(nn.Module):
    def __init__(self, model):
        super(ModelSoftmaxWrapper, self).__init__()
        self.model = model
    def forward(self, x, alpha=0.):
        out = self.model(x, alpha=alpha)
        out = torch.nn.functional.softmax(out, dim=1)
        return out

class EnsemblePredictor(nn.Module):
    def __init__(self, ensemble: list, num_classes=2, device='cpu'):
        super(EnsemblePredictor, self).__init__()
        self.device = device
        self.ensemble = ensemble
        self.num_classes = num_classes
    def forward(self, x, alpha=0.):
        out = torch.zeros(x.shape[0], self.num_classes).to(self.device)
        for model in self.ensemble:
            out += model(x, alpha=alpha)
        return out / len(self.ensemble)

