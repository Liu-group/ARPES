#################################################################
# code adapted from https://github.com/sleepyseal/TransferScore #
#################################################################
from utils.parsing import parse_args
from utils.utils import load_checkpoint, set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ARPESNet
from torch.utils.data import DataLoader
from dataset import ARPESDataset
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.random import uniform
import math
from random import sample
from torchvision import transforms
from torchvision.transforms import Normalize

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def get_transfer_score(target_dataset, model, num_classes, rank):
    im_loss=[]
    model.eval()
    if len(target_dataset)<20:
        ts_cluster = False
    else:
        ts_cluster = True
    ts_target_loader = DataLoader(target_dataset, batch_size=len(target_dataset), shuffle=True)
    train_target_iter = iter(ts_target_loader)
    iter_num = len(train_target_iter)
    assert iter_num == 1 # only one iteration implemented
    output_fs = []
    def hook(module, input, output):
        output_fs.append(input)
    model.class_classifier.register_forward_hook(hook)
    with torch.no_grad():
        data_target = train_target_iter.next()
        t_img, _ = data_target
        t_img = t_img.unsqueeze(1).to(rank)

        class_output, _ = model(t_img, alpha=1)

        softmax_out = nn.Softmax(dim=1)(class_output)
        entropy_loss = torch.mean(entropy(softmax_out))
        # print(entropy_loss)
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
        entropy_loss -= gentropy_loss

        im_loss.append(entropy_loss.item())

    if ts_cluster == True:
        X = output_fs[0][0].cpu().numpy()
        sample_size = int(X.shape[0]*0.05) #0.05 (5%) based on paper by Lawson and Jures

        #a uniform random sample in the original data space
        X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
        random_indices=sample(range(0, X.shape[0], 1), sample_size)
        X_sample = X[random_indices]

        #initialise unsupervised learner for implementing neighbor searches
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs=neigh.fit(X)
        
        u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
        u_distances = u_distances[: , 0] 
        
        w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
        w_distances = w_distances[: , 1]
        
        u_sum = np.sum(u_distances)
        w_sum = np.sum(w_distances)
        H = u_sum/ (u_sum + w_sum)
        if math.isnan(H):
            H=0
    else:
        H=0
    score= H-sum(im_loss)/len(im_loss)/math.log(num_classes)
    del ts_target_loader, train_target_iter
    return score