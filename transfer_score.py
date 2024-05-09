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
    
if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load exp data
    num_classes = args.num_classes
    data_path = args.data_path
    X_2014 = np.load(data_path + '/data_' + 'exp_2014' + '_processed_nobg.npy')
    y_2014 = np.load(data_path + '/data_' + 'exp_2014' + '_sc_values.npy').astype(int)
    X_2015 = np.load(data_path + '/data_' + 'exp_2015' + '_processed_nobg.npy')
    y_2015 = np.load(data_path + '/data_' + 'exp_2015' + '_sc_values.npy').astype(int)
    if num_classes==2:
        y_2014[y_2014==1]=0
        y_2014[y_2014==2]=1
        y_2015[y_2015==1]=0
        y_2015[y_2015==2]=1
    # dataloader for exp data
    transform_2014 = transforms.Compose([transforms.ToTensor(),
                                    Normalize((1.000,), (1.754))])
    transform_2015 = transforms.Compose([transforms.ToTensor(),
                                    Normalize((1.000,), (1.637))])

    target_2014 = ARPESDataset(X_2014, transform=transform_2014)
    target_2015 = ARPESDataset(X_2015, transform=transform_2015)
    target_dataset = torch.utils.data.ConcatDataset([target_2014, target_2015])
    target_loader = DataLoader(target_dataset, batch_size=len(y_2014)+len(y_2015), shuffle=True)

    tcs = []
    init_seed = args.seed
    for fold_num in range(args.num_folds):
        print(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        set_seed(args.seed)
        model = load_checkpoint(args).to(device)
        train_target_iter = iter(target_loader)
        tc = get_transfer_score(train_target_iter, model, args.num_classes, device)
        tcs.append(tc)
        print(f'Transferability Coefficient: {tc:.3f}')
    print('Transferability Coefficient: {:.3f} ± {:.3f}'.format(np.mean(tcs), np.std(tcs)))