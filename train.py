from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.utils import load_checkpoint, normalize_transform, entropy, ForeverDataIterator
from dataset import ARPESDataset
from torch.utils.data import DataLoader
import collections
from visualize import visualize
from transfer_score import get_transfer_score

def get_data_split(args, y):
    idx_all = np.arange(len(y))
    train_ratio, val_ratio, test_ratio = args.split[0], args.split[1], args.split[2] 

    idx_train, idx_test = train_test_split(idx_all, test_size=test_ratio, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, train_size=train_ratio/(1-test_ratio) , test_size=val_ratio/(1-test_ratio), random_state=42, stratify=y[idx_train])

    return idx_train, idx_val, idx_test

def run_training(args, model, data_source, data_target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_source, y_source = data_source
    X_target, _ = data_target
    idx_train, idx_val, idx_test = get_data_split(args, y_source)
    print("Number of sim training samples: ", len(idx_train))
    # source data loader
    source_dataset = ARPESDataset(X_source, y_source, transform=normalize_transform('sim'))
    train_dataset = torch.utils.data.Subset(source_dataset, idx_train)
    val_dataset = torch.utils.data.Subset(source_dataset, idx_val)
    test_dataset = torch.utils.data.Subset(source_dataset, idx_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(idx_val), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(idx_test), shuffle=False)

    # target data loader
    target_dataset = ARPESDataset(X_target, transform=normalize_transform(args.adv_on))
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
    print('Target data size: ', len(target_dataset))
    
    loss_func = nn.CrossEntropyLoss()
    
    metric_func = accuracy_score
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss, best_epoch, best_score = float('inf'), 0, -float('inf')
    model = model.to(device)
    for epoch in range(args.epochs):
        train_losses = train(args, epoch, model, train_loader, target_loader, loss_func, optimizer, device)
        val_losses, val_score = evaluate(model, val_loader, target_loader, loss_func, metric_func, device)

        if args.opt_goal=='ts':
            ts_model = copy.deepcopy(model)
            ts = get_transfer_score(target_dataset, ts_model, args.num_classes, device)
            del ts_model
        else:
            ts = 0
        print(f"Epoch: {epoch:02d} | Train Label Loss: {train_losses['err_s_label']:.3f} | Train Domain Loss: {train_losses['err_domain']:.3f} | Val Loss: {val_losses:.3f} | Val Acc: {val_score:.3f} | ts: {ts:.3f}")

        if args.opt_goal=='accuracy' and val_score > best_score \
        or args.opt_goal=='val_loss' and val_losses < best_val_loss \
        or args.opt_goal=='ts' and ts > best_score:
            best_score = val_score if args.opt_goal=='accuracy' else ts
            best_val_loss = val_losses
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        if epoch - best_epoch > args.early_stop_epoch:
            break       
    print(f"Best Epoch: {best_epoch:02d} | Best {args.opt_goal}: {best_score:.3f}")
    # save
    if args.save_best_model:
        save_path = os.path.join(args.checkpoint_path, str(args.num_classes) + '_' + str(args.seed) + f'_model_{args.adaptation}.pt')
        torch.save({"full_model": best_model, "state_dict": best_model.state_dict(), "args": args}, save_path)
        print('Saved the best model as ' + save_path)

    # visualize
    if args.visualize:
        vis_model_0 = copy.deepcopy(best_model)
        visualize(args, vis_model_0)
    
    test_losses, test_score, y_true, y_pred = predict(model, test_loader, loss_func, metric_func, device)
    print(f'Test Loss: {test_losses:.3f} | Test Acc: {test_score:.3f}')
    print(classification_report(y_true, y_pred, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1'], zero_division=0))
    print(confusion_matrix(y_true, y_pred))

    return best_model, test_score, best_score

def train(args, epoch, model, train_loader, target_loader, loss_func, optimizer, rank):
    model.train()
    if args.few == False:
        len_dataloader = min(len(train_loader), len(target_loader))
    else:
        len_dataloader = len(train_loader)
    data_source_iter = ForeverDataIterator(train_loader)
    data_target_iter = ForeverDataIterator(target_loader)
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
        s_domain_label = torch.zeros(batch_size).long().to(rank)
        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        
        s_class_output, s_domain_output = model(input_img, alpha)
        err_s_label = loss_func(s_class_output, class_label)
        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target
        t_img = t_img.unsqueeze(1).to(rank)
        batch_size = len(t_img)

        input_img = torch.Tensor(batch_size, 1, 400, 195).to(rank)
        t_domain_label = torch.ones(batch_size).long().to(rank)
        input_img.resize_as_(t_img).copy_(t_img).to(rank)

        t_class_output, t_domain_output = model(input_img, alpha)
        domain_output = torch.cat((s_domain_output, t_domain_output), dim=0)
        domain_label = torch.cat((s_domain_label, t_domain_label), dim=0)
        if args.entropy_conditioning == True:
            g = F.softmax(torch.cat((s_class_output, t_class_output), dim=0), dim=1).detach()
            weight = 1.0 + torch.exp(-entropy(g))
            weight = weight / torch.sum(weight) * len(weight)
            err_domain = nn.CrossEntropyLoss(reduction='none')(domain_output, domain_label)
            err_domain = torch.mean(err_domain * weight)
        else: 
            err_domain = loss_func(domain_output, domain_label)
        err = args.adaptation * err_domain + err_s_label
        err.backward()
        optimizer.step()
        i += 1
        total_losses['err_all'] += err.item()
        total_losses['err_s_label'] += err_s_label.item()
        total_losses['err_domain'] += err_domain.item()

    total_losses['err_all'] = total_losses['err_all']/len_dataloader
    total_losses['err_s_label'] = total_losses['err_s_label']/len_dataloader
    total_losses['err_domain'] = total_losses['err_domain']/len_dataloader
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
            X, _ = data_target
            X = X.unsqueeze(1).double().to(rank)
            batch_size = X.size(0)
            domain_label = torch.ones(batch_size).long().to(rank)
            class_output, domain_output = model(X, alpha=1)
            loss = loss_func(domain_output, domain_label)
            target_loss_total += loss.item() * X.size(0)
            target_count += X.size(0)

        val_losses = source_loss_total/source_count + target_loss_total/target_count
        val_score = metric_func(y_pred, y_true)

    return val_losses, val_score

def predict(model, test_loader, loss_func, metric_func, rank, prob=False):
    model.eval()
    model.prediction_mode = True
    y_pred, y_true, prob = [], [], []
    count, loss_total = 0, 0.0
    with torch.no_grad():
        for data in test_loader:
            X, y, _ = data
            X, y = X.unsqueeze(1).double().to(rank), y.long().to(rank)
            out = model(X, alpha=0)
            loss = loss_func(out, y)
            prob.append(out.cpu().numpy())
            pred = torch.argmax(out, dim=1)
            loss_total += loss.item() * X.size(0)
            count += X.size(0)
            y_pred += pred.tolist()
            y_true += y.tolist()            
        test_losses = loss_total/count
        test_score = metric_func(y_pred, y_true)
    
    if prob==True:
        prob = np.concatenate(prob)
        return test_losses, test_score, y_true, prob
    else:
        return test_losses, test_score, y_true, y_pred
