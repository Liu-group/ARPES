
import numpy as np
import copy
import torch
from utils.parsing import parse_args
from utils.utils import load_checkpoint, set_seed   
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn as nn
from model import ARPESNet
from train import run_training, predict
from dataset import ARPESDataset
from typing import Optional
torch.set_default_tensor_type(torch.DoubleTensor)

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
        tss = []
        exp_2014_pred_all, exp_2015_pred_all = [], []
        exp_2014_acc_all, exp_2015_acc_all = [], []
        for fold_num in range(args.num_folds):
            print(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            set_seed(args.seed)
            X_source, y_source, _ = load_data(args.data_path, 'sim', args.num_classes)
            X_2014, y_2014, _ = load_data(args.data_path, 'exp_2014', args.num_classes)
            X_2015, y_2015, _ = load_data(args.data_path, 'exp_2015', args.num_classes)
            model = ARPESNet(num_classes=args.num_classes,
                            hidden_channels=args.hidden_channels, 
                            negative_slope=args.negative_slope,
                            dropout=args.dropout)
            best_model, _, ts = run_training(args, model, (X_source, y_source), (X_2014, X_2015))
            # y_exp_2014, y_exp_2015 are always the same at each fold
            y_pred_2014, y_exp_2014 = test_exp(args, 'exp_2014', best_model)
            y_pred_2015, y_exp_2015 =  test_exp(args, 'exp_2015', best_model)
            exp_2014_pred_all.append(y_pred_2014)
            exp_2015_pred_all.append(y_pred_2015)
            tss.append(ts)
            exp_2014_acc_all.append(accuracy_score(y_exp_2014, y_pred_2014))
            exp_2015_acc_all.append(accuracy_score(y_exp_2015, y_pred_2015))
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
        print('Exp_2014 Accuracy: {:.3f} ± {:.3f}'.format(np.mean(exp_2014_acc_all), np.std(exp_2014_acc_all)))
        print('Exp_2015 Accuracy: {:.3f} ± {:.3f}'.format(np.mean(exp_2015_acc_all), np.std(exp_2015_acc_all)))
        print('Transfer Score: {:.3f} ± {:.3f}'.format(np.mean(tss), np.std(tss)))
        print(model)
    if args.mode == 'adv_train':
        set_seed(args.seed)
        X_source, y_source, _ = load_data(args.data_path, 'sim', args.num_classes)
        X_2014, y_2014, _ = load_data(args.data_path, 'exp_2014', args.num_classes)
        X_2015, y_2015, _ = load_data(args.data_path, 'exp_2015', args.num_classes)
        model = ARPESNet(num_classes=args.num_classes,
                         hidden_channels=args.hidden_channels, 
                         negative_slope=args.negative_slope,
                         dropout=args.dropout)
        best_model, _, _ = run_training(args, model, (X_source, y_source), (X_2014, X_2015))
        test_exp(args, 'exp_2014', best_model)
        test_exp(args, 'exp_2015', best_model)
        print(model)
    if args.mode == 'predict':
        set_seed(args.seed)
        test_exp(args, 'exp_2014')
        test_exp(args, 'exp_2015')

if __name__ == '__main__':
    main()