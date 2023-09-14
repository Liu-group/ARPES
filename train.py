from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def run_training(args, model, train_loader, val_loader, test_loader):
    loss_func = nn.CrossEntropyLoss()
    metric_func = accuracy_score#Accuracy(task='multiclass' if args.num_classes>2 else 'binary', num_classes=args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_score = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(args.epochs):
        train_losses = train(model, train_loader, loss_func, optimizer, device)
        val_losses, val_score = evaluate(model, val_loader, loss_func, metric_func, device)
        print(f'Epoch: {epoch:02d} | Train Loss: {train_losses:.3f} | Val Loss: {val_losses:.3f} | Val Acc: {val_score:.3f}')
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_model = copy.deepcopy(model)

    print(f'Best Epoch: {best_epoch:02d} | Val Loss: {best_score:.3f}')

    # save
    if args.save_best_model:
        torch.save({"full_model": best_model, "state_dict": model.state_dict(), "args": args}, args.checkpoint_path)
        print('Saved the best model to ' + args.checkpoint_path)
    test_losses, test_score, y_true, y_pred = predict(model, test_loader, loss_func, metric_func, device)
    print(f'Test Loss: {test_losses:.3f} | Test Acc: {test_score:.3f}')
    print(classification_report(y_true, y_pred, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
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
            pred = torch.argmax(out, dim=1)
            loss_total += loss.item() * X.size(0)
            count += X.size(0)
            y_pred += pred.tolist()
            y_true += y.tolist()
        val_losses = loss_total/count
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