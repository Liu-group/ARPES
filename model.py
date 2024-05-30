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
                 fcw=64,  
                 kernel_size=3,
                 prediction_mode = False,
                 conditional = False,
                 pool_layer = True,
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
            Conv2d(in_channels=hidden_channels*2, out_channels=hidden_channels*4, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),
            Conv2d(in_channels=hidden_channels*4, out_channels=hidden_channels*4, kernel_size=kernel_size, stride=1, padding=2),
            MaxPool2d(kernel_size=kernel_size, stride=2),
            ReLU(),
            )

        self.prediction_mode = prediction_mode
        self.dropout = Dropout(dropout)
        x = torch.empty(1, 1, 400, 195)
        out = self.convs(x)
        self.pool_layer = pool_layer
        if pool_layer:
            h = int((fcw*2/(hidden_channels//2))**0.5)
            self.pool_layer = nn.AdaptiveAvgPool2d(output_size=(5, 5))
            out = self.pool_layer(out)
        h_shape = out.size()[1:]
        #h_shape: torch.Size([32, 50, 25])
        out = out.view(1, -1)
        out_dim = out.size(-1)

        self.domain_classifier = Sequential(
            Linear(out_dim if conditional==False else out_dim*num_classes, fcw),
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
        self.conditional = conditional
        self.map = MultiLinearMap()

    def forward(self, x, alpha=1.):
        f = self.convs(x)
        if self.pool_layer:
            f = self.pool_layer(f)
        f = f.view(f.size(0), -1)
        f = self.dropout(f)
        class_out = self.class_classifier(f)    
        if self.conditional == True:
            out = F.softmax(class_out, dim=1).detach()
            f_domain = self.map(f, out)
        else:
            f_domain = f
        reverse_x = ReverseLayerF.apply(f_domain, alpha)
        domain_out = self.domain_classifier(reverse_x)
        
        if self.prediction_mode:
            return class_out
        else:
            return class_out, domain_out

class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)

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

