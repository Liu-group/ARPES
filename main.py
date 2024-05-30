
import numpy as np
import copy
import torch
from utils.parsing import parse_args
from utils.utils import load_checkpoint, set_seed, normalize_transform, get_num_sample
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn as nn
from model import ARPESNet, ModelSoftmaxWrapper, EnsemblePredictor
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

def test_exp(args, x_exp, y_exp, name, model: Optional[nn.Module] = None):
    """ test on exp data """
    # dataloader for exp data
    show_prob = True if model != None else False
    exp_dataset = ARPESDataset(x_exp, y_exp, transform=normalize_transform(name))
    exp_loader = DataLoader(exp_dataset, batch_size=len(y_exp), shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = load_checkpoint(args).to(device)
    else:
        model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    metric_func = accuracy_score
    test_losses, test_score, y_true, y_pred = predict(model, exp_loader, loss_func, metric_func, device, show_prob=show_prob)
    print(f'{name} Test Loss: {test_losses:.3f} | {name} Test Acc: {test_score:.3f}')
    return list(y_pred), list(y_true)

def main():
    args = parse_args()
    print(args)
    if args.mode == 'train':
        init_seed = args.seed
        tss = []
        tss, pred_all, acc_all = [], [], []
        X_source, y_source, _ = load_data(args.data_path, 'sim', args.num_classes)
        if args.adv_on == 'exp_all':
            X_2014, y_2014, _ = load_data(args.data_path, 'exp_2014', args.num_classes)
            X_2015, y_2015, _ = load_data(args.data_path, 'exp_2015', args.num_classes)
            X_target = np.concatenate((X_2014, X_2015), axis=0)
            y_target = np.concatenate((y_2014, y_2015), axis=0)
            if args.num_adv != 86:
                X_target, y_target = get_num_sample(X_target, y_target, args.num_adv)
        else:
            X_target, y_target, _ = load_data(args.data_path, args.adv_on, args.num_classes)
            if (args.num_adv != 41 and args.adv_on == 'exp_2015') or (args.num_adv != 45 and args.adv_on == 'exp_2014'):
                X_target, y_target = get_num_sample(X_target, y_target, args.num_adv)        
        for fold_num in range(args.num_folds):
            print(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            set_seed(args.seed)
            model = ARPESNet(num_classes=args.num_classes,
                            hidden_channels=args.hidden_channels, 
                            negative_slope=args.negative_slope,
                            dropout=args.dropout,
                            conditional=args.conditional,
                            pool_layer=args.pool_layer,
                            fcw=args.fcw,
                            )
            best_model, _, ts = run_training(args, model, (X_source, y_source), (X_target, _))
            pred, y = test_exp(args, X_target, y_target, args.adv_on, best_model)
            pred_all.append(pred)
            tss.append(ts)
            acc_all.append(accuracy_score(y, pred))
        # get mean and std of accuracy, precision, recall, f1-score
        pred_all = np.array(pred_all)
        print('{} Accuracy: {:.3f} ± {:.3f}'.format(args.adv_on, np.mean(acc_all), np.std(acc_all)))
        print('{} Transfer Score: {:.3f} ± {:.3f}'.format(args.adv_on, np.mean(tss), np.std(tss)))
        print(model)
        
    if args.mode == 'predict':
        set_seed(args.seed)
        ensemble = []
        exp_2014_acc_all, exp_2015_acc_all = [], []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(10):
            args.seed = 42 + i
            X_2014, y_2014, _ = load_data(args.data_path, 'exp_2014', args.num_classes)
            X_2015, y_2015, _ = load_data(args.data_path, 'exp_2015', args.num_classes)
            pred_2014, exp_2014 = test_exp(args, X_2014, y_2014, 'exp_2014')
            pred_2015, exp_2015 =  test_exp(args, X_2015, y_2015, 'exp_2015')
            exp_2014_acc_all.append(accuracy_score(exp_2014, pred_2014))
            exp_2015_acc_all.append(accuracy_score(exp_2015, pred_2015))
            model = load_checkpoint(args).to(device)
            softmax_model = ModelSoftmaxWrapper(model)
            ensemble.append(softmax_model)
        ensemble_predictor = EnsemblePredictor(ensemble, num_classes=args.num_classes, device=device)
        y_pred_2014, y_exp_2014 = test_exp(args, X_2014, y_2014, 'exp_2014',  ensemble_predictor)
        y_pred_2015, y_exp_2015 =  test_exp(args, X_2015, y_2015, 'exp_2015', ensemble_predictor)
        print('Exp_2014 Accuracy: {:.3f} ± {:.3f}'.format(np.mean(exp_2014_acc_all), np.std(exp_2014_acc_all)))
        print('Exp_2015 Accuracy: {:.3f} ± {:.3f}'.format(np.mean(exp_2015_acc_all), np.std(exp_2015_acc_all)))
        print('Exp_2014 Ensemble')
        print(classification_report(y_exp_2014, y_pred_2014, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2014, y_pred_2014))
        print('Exp_2015 Ensemble')
        print(classification_report(y_exp_2015, y_pred_2015, target_names=['0', '1', '2'] if args.num_classes==3 else ['0', '1']))
        print(confusion_matrix(y_exp_2015, y_pred_2015))
if __name__ == '__main__':
    main()