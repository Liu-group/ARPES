
import numpy as np
import random
import copy
import torch
from utils.parsing import parse_args
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from dataset import ARPESDataset
from model import ARPESNet
from train import run_training, predict
from torchvision import transforms
from torchvision.transforms import RandomAffine, RandomRotation, Normalize
from utils.utils import AddGaussianNoise, load_checkpoint
from sklearn.preprocessing import OneHotEncoder
torch.set_default_tensor_type(torch.DoubleTensor)

def set_seed(seed=42):
    """Sets initial seed for random numbers."""
    # set random seed for torch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_data(args):
    name = 'sim' if args.mode=='train' else args.dataset_name
    x = np.load(args.data_path + '/data_' + name + '_processed_nobg.npy')
    y = np.load(args.data_path + '/data_' + name + '_sc_values.npy')
    temp = np.load(args.data_path + '/data_' + name + '_templist.npy')
    
    if args.num_classes==2:
        y[y==1]=0
        y[y==2]=1
    return x, y, temp

def get_data_split(args, y):
    idx_all = np.arange(len(y))
    test_ratio = args.split[2] if args.split[2]<1 else args.split[2]/len(y)
    val_ratio = args.split[1] if args.split[1]<1 else args.split[1]/len(y)

    idx_train, idx_test = train_test_split(idx_all, test_size=test_ratio, random_state=args.seed, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_ratio/(1-test_ratio), random_state=args.seed, stratify=y[idx_train])

    return idx_train, idx_val, idx_test
    
def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mode == 'train':
        X, y, temp = load_data(args)
        idx_train, idx_val, idx_test = get_data_split(args, y)
        transform = transforms.Compose([transforms.ToTensor(),
                                        #RandomRotation(degrees=(0, 90)),
                                        RandomAffine(degrees=0, 
                                                    translate=(args.trf,args.trf), 
                                                    scale=(args.scalef,  1/args.scalef),
                                                    interpolation=transforms.InterpolationMode.BILINEAR),
                                        AddGaussianNoise(0., 0.25)])
        train_dataset = ARPESDataset(X[idx_train], y[idx_train], transform=transform)
        val_dataset = ARPESDataset(X[idx_val], y[idx_val])
        test_dataset = ARPESDataset(X[idx_test], y[idx_test])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = ARPESNet(num_classes=args.num_classes)
        run_training(args, model, train_loader, val_loader, test_loader)

    if args.mode == 'predict':
        X, y, temp = load_data(args)
        dataset = ARPESDataset(X, y)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_checkpoint(args).to(device)
        loss_func = nn.CrossEntropyLoss()
        metric_func = accuracy_score
        test_losses, test_score, y_true, y_pred = predict(model, test_loader, loss_func, metric_func, device)
        print(f'Test Loss: {test_losses:.3f} | Test Acc: {test_score:.3f}')
        print(classification_report(y_true, y_pred, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
            

    

if __name__ == '__main__':
    main()