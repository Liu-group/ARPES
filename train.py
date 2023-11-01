from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import RandomAffine, RandomRotation, Normalize
from utils.utils import AddGaussianNoise
from dataset import ARPESDataset
from torch.utils.data import DataLoader

def get_data_split(args, y):
    idx_all = np.arange(len(y))
    test_ratio = args.split[2] if args.split[2]<1 else args.split[2]/len(y)
    val_ratio = args.split[1] if args.split[1]<1 else args.split[1]/len(y)

    idx_train, idx_test = train_test_split(idx_all, test_size=test_ratio, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_ratio/(1-test_ratio), random_state=42, stratify=y[idx_train])

    return idx_train, idx_val, idx_test

def run_training(args, model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = data
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
    
    num_labels = np.bincount(train_dataset.targets)
    weight = torch.tensor([(1 / i) * (num_labels.sum() / 2.0) for i in num_labels]).to(device)
    loss_func = nn.CrossEntropyLoss(weight=weight)
    
    metric_func = balanced_accuracy_score
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score, best_epoch = 0., 0
    model = model.to(device)
    for epoch in range(args.epochs):
        train_losses = train(model, train_loader, loss_func, optimizer, device)
        val_losses, val_score = evaluate(model, val_loader, loss_func, metric_func, device)
        print(f'Epoch: {epoch:02d} | Train Loss: {train_losses:.3f} | Val Loss: {val_losses:.3f} | Val Acc: {val_score:.3f}')
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        if epoch - best_epoch > args.early_stop_epoch:
            break       

    print(f'Best Epoch: {best_epoch:02d} | Val Loss: {best_score:.3f}')

    # save
    if args.save_best_model:
        save_path = os.path.join(args.checkpoint_path, str(args.seed) + '_model.pt')
        torch.save({"full_model": best_model, "state_dict": model.state_dict(), "args": args}, save_path)
        print('Saved the best model as ' + save_path)
    test_losses, test_score, y_true, y_pred = predict(model, test_loader, loss_func, metric_func, device)
    print(f'Test Loss: {test_losses:.3f} | Test Acc: {test_score:.3f}')
    print(classification_report(y_true, y_pred, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
    print(confusion_matrix(y_true, y_pred))
    return test_score

def train(model, train_loader, loss_func, optimizer, rank):
    model.train()
    count, loss_total = 0, 0.0
    for data in train_loader:
        X, y, _ = data
        X, y = X.unsqueeze(1).double().to(rank), y.long().to(rank)
        out = model(X)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item() * X.size(0)
        count += X.size(0)
    train_losses = loss/count

    return train_losses

def evaluate(model, val_loader, loss_func, metric_func, rank):
    model.eval()
    y_pred, y_true = [], []
    count, loss_total = 0, 0.0
    with torch.no_grad():
        for data in val_loader:
            X, y, _ = data
            X, y = X.unsqueeze(1).double().to(rank), y.long().to(rank)
            out = model(X)
            loss = loss_func(out, y)
            print("out:",out)
            pred = torch.argmax(out, dim=1)
            loss_total += loss.item() * X.size(0)
            count += X.size(0)
            y_pred += pred.tolist()
            y_true += y.tolist()
        val_losses = loss_total/count
        val_score = metric_func(y_pred, y_true)
        print(y_pred)
        print(y_true)
    return val_losses, val_score

def predict(model, test_loader, loss_func, metric_func, rank):
    model.eval()
    y_pred, y_true = [], []
    count, loss_total = 0, 0.0
    with torch.no_grad():
        for data in test_loader:
            X, y, _ = data
            X, y = X.unsqueeze(1).double().to(rank), y.long().to(rank)
            out = model(X)
            loss = loss_func(out, y)
            pred = torch.argmax(out, dim=1)
            loss_total += loss.item() * X.size(0)
            count += X.size(0)
            y_pred += pred.tolist()
            y_true += y.tolist()            
        test_losses = loss_total/count
        test_score = metric_func(y_pred, y_true)
    return test_losses, test_score, y_true, y_pred