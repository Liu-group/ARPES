from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import RandomAffine, RandomRotation, Normalize
from utils.utils import AddGaussianNoise
from dataset import ARPESDataset
from torch.utils.data import DataLoader
import collections
from visualize import visualize

def get_data_split(args, y):
    idx_all = np.arange(len(y))
    train_ratio, val_ratio, test_ratio = args.split[0], args.split[1], args.split[2] 

    idx_train, idx_test = train_test_split(idx_all, test_size=test_ratio, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_ratio, random_state=42, stratify=y[idx_train])

    return idx_train, idx_val, idx_test

def run_training(args, model, data_source, data_target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_source, y_source = data_source
    X_target, _ = data_target
    idx_train, idx_val, idx_test = get_data_split(args, y_source)
    transform = transforms.Compose([transforms.ToTensor(),
                                    #RandomRotation(degrees=(0, 90)),
                                    RandomAffine(degrees=0, 
                                                translate=(args.trf,args.trf), 
                                                scale=(args.scalef,  1/args.scalef),
                                                interpolation=transforms.InterpolationMode.BILINEAR),
                                    AddGaussianNoise(0., 0.25)])
    # source data loader
    source_dataset = ARPESDataset(X_source, y_source, transform=transform)
    train_dataset = torch.utils.data.Subset(source_dataset, idx_train)
    val_dataset = torch.utils.data.Subset(source_dataset, idx_val)
    test_dataset = torch.utils.data.Subset(source_dataset, idx_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(idx_val), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(idx_test), shuffle=False)

    # target data loader
    target_dataset = ARPESDataset(X_target, _, transform=transform)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
    
    #num_labels = np.bincount(train_dataset.targets)
    #weight = torch.tensor([(1 / i) * (num_labels.sum() / 2.0) for i in num_labels]).to(device)
    loss_func = nn.CrossEntropyLoss()#weight=weight)
    
    metric_func = accuracy_score#balanced_accuracy_score
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=args.min_lr)

    best_val_loss, best_epoch = 1e10, 0
    model = model.to(device)
    for epoch in range(args.epochs):
        train_losses = train(args, epoch, model, train_loader, target_loader, loss_func, optimizer, device)
        val_losses, val_score = evaluate(model, val_loader, target_loader, loss_func, metric_func, device)
        print(f"Epoch: {epoch:02d} | Train Label Loss: {train_losses['err_s_label']:.3f} | Train Domain Loss: {train_losses['err_s_domain']:.3f} | Train Target Domain Loss: {train_losses['err_t_domain']:.3f} | Val Loss: {val_losses:.3f} | Val Acc: {val_score:.3f}")
       
        if val_losses < best_val_loss:
            best_score = val_score
            best_val_loss = val_losses
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        if epoch - best_epoch > args.early_stop_epoch:
            break       
        scheduler.step(val_losses)

    print(f"Best Epoch: {best_epoch:02d} | Best Val Acc: {best_score:.3f}")
    # save
    if args.save_best_model:
        save_path = os.path.join(args.checkpoint_path, str(args.seed) + '_model.pt')
        torch.save({"full_model": best_model, "state_dict": model.state_dict(), "args": args}, save_path)
        print('Saved the best model as ' + save_path)

    # visualize
    if args.visualize:
        vis_model = copy.deepcopy(model)
        visualize(args, vis_model)

    test_losses, test_score, y_true, y_pred = predict(model, test_loader, loss_func, metric_func, device)
    print(f'Test Loss: {test_losses:.3f} | Test Acc: {test_score:.3f}')
    print(classification_report(y_true, y_pred, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
    print(confusion_matrix(y_true, y_pred))
    return test_score

def train(args, epoch, model, train_loader, target_loader, loss_func, optimizer, rank):
    model.train()
    len_dataloader = min(len(train_loader), len(target_loader))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(target_loader)
    total_losses = collections.defaultdict(float)
    i = 0
    while i < len_dataloader:
        p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label, _ = data_source
        s_img = s_img.unsqueeze(1).to(rank)
        s_label = s_label.long().to(rank)
        
        model.zero_grad()
        batch_size = len(s_label)

        input_img = torch.Tensor(batch_size, 1, 400, 195).to(rank)
        class_label = torch.LongTensor(batch_size).to(rank)
        domain_label = torch.zeros(batch_size).long().to(rank)
        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        
        class_output, domain_output = model(input_img, alpha)
        err_s_label = loss_func(class_output, class_label)
        err_s_domain = loss_func(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _, _ = data_target
        t_img = t_img.unsqueeze(1).to(rank)
        batch_size = len(t_img)

        input_img = torch.Tensor(batch_size, 1, 400, 195).to(rank)
        domain_label = torch.ones(batch_size).long().to(rank)
        input_img.resize_as_(t_img).copy_(t_img).to(rank)

        _, domain_output = model(input_img, alpha)
        err_t_domain = loss_func(domain_output, domain_label)
        err = args.adaptation*(err_t_domain + err_s_domain) + err_s_label
        err.backward()
        optimizer.step()
        i += 1
        total_losses['err_all'] += err.cpu().data.numpy()
        total_losses['err_s_label'] += err_s_label.cpu().data.numpy()
        total_losses['err_s_domain'] += err_s_domain.cpu().data.numpy()
        total_losses['err_t_domain'] += err_t_domain.cpu().data.numpy()

    total_losses['err_all'] = total_losses['err_all']/len_dataloader
    total_losses['err_s_label'] = total_losses['err_s_label']/len_dataloader
    total_losses['err_s_domain'] = total_losses['err_s_domain']/len_dataloader
    total_losses['err_t_domain'] = total_losses['err_t_domain']/len_dataloader
    return total_losses

def evaluate(model, val_loader, target_loader, loss_func, metric_func, rank):
    model.eval()
    y_pred, y_true = [], []
    source_count, target_count = 0, 0
    source_loss_total, target_loss_total = 0, 0.0
    with torch.no_grad():
        for data_source in val_loader:
            X, y, _ = data_source
            X, y = X.unsqueeze(1).double().to(rank), y.long().to(rank)
            batch_size = len(y)
            domain_label = torch.zeros(batch_size).long().to(rank)
            class_label = torch.LongTensor(batch_size).to(rank)
            class_output, domain_output = model(X, alpha=1)

            class_loss = loss_func(class_output, y)
            domain_loss = loss_func(domain_output, domain_label)

            pred = torch.argmax(class_output, dim=1)
            source_loss_total += (class_loss.item() + domain_loss.item()) * X.size(0)
            source_count += X.size(0)
            y_pred += pred.tolist()
            y_true += y.tolist()
        
        for data_target in target_loader:
            X, _, _ = data_target
            X = X.unsqueeze(1).double().to(rank)
            batch_size = X.size(0)
            domain_label = torch.ones(batch_size).long().to(rank)
            _, domain_output = model(X, alpha=1)
            loss = loss_func(domain_output, domain_label)
            target_loss_total += loss.item() * X.size(0)
            target_count += X.size(0)

        val_losses = source_loss_total/source_count + target_loss_total/target_count
        val_score = metric_func(y_pred, y_true)

    return val_losses, val_score

def predict(model, test_loader, loss_func, metric_func, rank):
    model.eval()
    y_pred, y_true = [], []
    count, loss_total = 0, 0.0
    with torch.no_grad():
        for data in test_loader:
            X, y, _ = data
            X, y = X.unsqueeze(1).double().to(rank), y.long().to(rank)
            out, _ = model(X, alpha=0)
            loss = loss_func(out, y)
            pred = torch.argmax(out, dim=1)
            loss_total += loss.item() * X.size(0)
            count += X.size(0)
            y_pred += pred.tolist()
            y_true += y.tolist()            
        test_losses = loss_total/count
        test_score = metric_func(y_pred, y_true)
    return test_losses, test_score, y_true, y_pred