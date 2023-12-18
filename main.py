
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
from typing import Optional

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

def test_exp(args, name, model: Optional[nn.Module] = None):
    """ test on exp data """
    # load exp data
    x_exp, y_exp, _ = load_data(args.data_path, name, args.num_classes)
    # dataloader for exp data
    exp_dataset = ARPESDataset(x_exp, y_exp)
    exp_loader = DataLoader(exp_dataset, batch_size=len(y_exp), shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = load_checkpoint(args).to(device)
    loss_func = nn.CrossEntropyLoss()
    metric_func = accuracy_score
    test_losses, test_score, y_true, y_pred = predict(model, exp_loader, loss_func, metric_func, device)
    print(f'{name} Test Loss: {test_losses:.3f} | {name} Test Acc: {test_score:.3f}')
    return list(y_pred), list(y_true)

def main():
    args = parse_args()
    print(args)
    if args.mode == 'cross_val_adv_train':
        init_seed = args.seed
        exp_2014_pred_all, exp_2015_pred_all = [], []
        for fold_num in range(args.num_folds):
            print(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            set_seed(args.seed)
            X_source, y_source, _ = load_data(args.data_path, 'sim', args.num_classes)
            X_target, y_target, _ = load_data(args.data_path, args.adv_on, args.num_classes)
            model = ARPESNet(num_classes=args.num_classes, dropout=args.dropout)
            run_training(args, model, (X_source, y_source), (X_target, _))
            # y_exp_2014, y_exp_2015 are always the same at each fold
            y_pred_2014, y_exp_2014 = test_exp(args, 'exp_2014', model)
            y_pred_2015, y_exp_2015 =  test_exp(args, 'exp_2015', model)
            exp_2014_pred_all.append(y_pred_2014)
            exp_2015_pred_all.append(y_pred_2015)
        # get mean and std of accuracy, precision, recall, f1-score
        exp_2014_pred_all = np.array(exp_2014_pred_all)
        print("exp_2014_pred_all.shape",exp_2014_pred_all.shape)
        exp_2015_pred_all = np.array(exp_2015_pred_all)
        print("exp_2015_pred_all.shape",exp_2015_pred_all.shape)
        print('Exp_2014')
        print(classification_report(y_exp_2014, np.mean(exp_2014_pred_all, axis=0).round(), target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2014, np.mean(exp_2014_pred_all, axis=0).round()))
        print('Exp_2015')
        print(classification_report(y_exp_2015, np.mean(exp_2015_pred_all, axis=0).round(), target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2015, np.mean(exp_2015_pred_all, axis=0).round()))
        print(model)
    if args.mode == 'adv_train':
        set_seed(args.seed)
        X_source, y_source, _ = load_data(args.data_path, 'sim', args.num_classes)
        X_target, y_target, _ = load_data(args.data_path, args.adv_on, args.num_classes)
        model = ARPESNet(num_classes=args.num_classes, dropout=args.dropout)
        run_training(args, model, (X_source, y_source), (X_target, _))
        test_exp(args, 'exp_2014')
        test_exp(args, 'exp_2015')
    if args.mode == 'predict':
        set_seed(args.seed)
        test_exp(args, 'exp_2014')
        test_exp(args, 'exp_2015')

if __name__ == '__main__':
    main()