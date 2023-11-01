
import numpy as np
import random
import copy
import torch
from utils.parsing import parse_args
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn as nn
from model import ARPESNet
from train import run_training, predict
from dataset import ARPESDataset
from utils.utils import load_checkpoint



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

def load_data(data_path, name, num_classes):
    x = np.load(data_path + '/data_' + name + '_processed_nobg.npy')
    y = np.load(data_path + '/data_' + name + '_sc_values.npy').astype(int)
    temp = np.load(data_path + '/data_' + name + '_templist.npy')
    if num_classes==2:
        y[y==1]=0
        y[y==2]=1
    return x, y, temp

    
def main():
    args = parse_args()
    if args.mode == 'train':
        set_seed(args.seed)
        X, y, _ = load_data(args.data_path, 'sim', args.num_classes)
        model = ARPESNet(num_classes=args.num_classes, dropout=args.dropout)
        run_training(args, model, (X, y))
        # test on exp data
        # load exp data
        x_exp_2014, y_exp_2014, _ = load_data(args.data_path, 'exp_2014', args.num_classes)
        x_exp_2015, y_exp_2015, _ = load_data(args.data_path, 'exp_2015', args.num_classes)
        # dataloader for exp data
        exp_2014_pred_all, exp_2015_pred_all = [], []
        exp_2014_dataset = ARPESDataset(x_exp_2014, y_exp_2014)
        exp_2015_dataset = ARPESDataset(x_exp_2015, y_exp_2015)
        exp_2014_loader = DataLoader(exp_2014_dataset, batch_size=len(y_exp_2014))
        exp_2015_loader = DataLoader(exp_2015_dataset, batch_size=len(y_exp_2015))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_checkpoint(args).to(device)
        loss_func = nn.CrossEntropyLoss()
        metric_func = accuracy_score
        test_losses, test_score, y_true, y_pred = predict(model, exp_2014_loader, loss_func, metric_func, device)
        print(f'Exp_2014 Test Loss: {test_losses:.3f} | Exp_2014 Test Acc: {test_score:.3f}')
        exp_2014_pred_all.append(list(y_pred))
        test_losses, test_score, y_true, y_pred = predict(model, exp_2015_loader, loss_func, metric_func, device)
        print(f'Exp_2015 Test Loss: {test_losses:.3f} | Exp_2015 Test Acc: {test_score:.3f}')
        exp_2015_pred_all.append(list(y_pred))
        # get mean and std of accuracy, precision, recall, f1-score
        exp_2014_pred_all = np.array(exp_2014_pred_all)
        print("exp_2014_pred_all.shape",exp_2014_pred_all.shape)
        exp_2015_pred_all = np.array(exp_2015_pred_all)
        print('Exp_2014')
        print(classification_report(y_exp_2014, np.mean(exp_2014_pred_all, axis=0).round(), target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2014, np.mean(exp_2014_pred_all, axis=0).round()))
        print('Exp_2015')
        print(classification_report(y_exp_2015, np.mean(exp_2015_pred_all, axis=0).round(), target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2015, np.mean(exp_2015_pred_all, axis=0).round()))

    if args.mode == 'predict':
        set_seed(args.seed)
        X, y, temp = load_data(args.data_path, args.dataset_name, args.num_classes)
        dataset = ARPESDataset(X, y)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_checkpoint(args).to(device)
        loss_func = nn.CrossEntropyLoss()
        metric_func = accuracy_score
        test_losses, test_score, y_true, y_pred = predict(model, test_loader, loss_func, metric_func, device)
        print(f'Test Loss: {test_losses:.3f} | Test Acc: {test_score:.3f}')
        print(classification_report(y_true, y_pred, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
    if args.mode == 'cross_validation_predict':
        args = parse_args()
        init_seed = args.seed
        exp_2014_pred_all, exp_2015_pred_all = [], []
        for fold_num in range(args.num_folds):
            print(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            set_seed(args.seed)
            X, y, _ = load_data(args.data_path, 'sim', args.num_classes)
            model = ARPESNet(num_classes=args.num_classes, dropout=args.dropout)
            run_training(args, model, (X, y))
            # test on exp data
            # load exp data
            x_exp_2014, y_exp_2014, _ = load_data(args.data_path, 'exp_2014', args.num_classes)
            x_exp_2015, y_exp_2015, _ = load_data(args.data_path, 'exp_2015', args.num_classes)
            # dataloader for exp data
            exp_2014_dataset = ARPESDataset(x_exp_2014, y_exp_2014)
            exp_2015_dataset = ARPESDataset(x_exp_2015, y_exp_2015)
            exp_2014_loader = DataLoader(exp_2014_dataset, batch_size=len(y_exp_2014))
            exp_2015_loader = DataLoader(exp_2015_dataset, batch_size=len(y_exp_2015))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = load_checkpoint(args).to(device)
            loss_func = nn.CrossEntropyLoss()
            metric_func = accuracy_score
            test_losses, test_score, y_true, y_pred = predict(model, exp_2014_loader, loss_func, metric_func, device)
            print(f'Exp_2014 Test Loss: {test_losses:.3f} | Exp_2014 Test Acc: {test_score:.3f}')
            exp_2014_pred_all.append(list(y_pred))
            test_losses, test_score, y_true, y_pred = predict(model, exp_2015_loader, loss_func, metric_func, device)
            print(f'Exp_2015 Test Loss: {test_losses:.3f} | Exp_2015 Test Acc: {test_score:.3f}')
            exp_2015_pred_all.append(list(y_pred))
        # get mean and std of accuracy, precision, recall, f1-score
        exp_2014_pred_all = np.array(exp_2014_pred_all)
        print("exp_2014_pred_all.shape",exp_2014_pred_all.shape)
        exp_2015_pred_all = np.array(exp_2015_pred_all)
        print('Exp_2014')
        print(classification_report(y_exp_2014, np.mean(exp_2014_pred_all, axis=0).round(), target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2014, np.mean(exp_2014_pred_all, axis=0).round()))
        print('Exp_2015')
        print(classification_report(y_exp_2015, np.mean(exp_2015_pred_all, axis=0).round(), target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2015, np.mean(exp_2015_pred_all, axis=0).round()))

if __name__ == '__main__':
    main()