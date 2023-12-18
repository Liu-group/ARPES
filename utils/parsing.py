import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='ARPESdata')
    parser.add_argument('--save_best_model', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='./models/')
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--mode', type=str, default='cross_val_adv_train') # predict, adv_train, cross_val_adv_train
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dataset_name', type=str, default='exp_2015')
    parser.add_argument('--adv_on', type=str, default='exp_2014', help="target dataset")
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--split', type=list, default=[0.8, 0.1, 0.1])
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--visualize', type=bool, default=True)

    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=4000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--early_stop_epoch", type=int, default=30, help="early stop epoch")
    parser.add_argument('--factor', type=float, default=0.9, help="factor for lr scheduler")
    parser.add_argument('--patience', type=int, default=10, help="patience for lr scheduler")
    parser.add_argument('--min_lr', type=int, default=1e-7, help="minimum learning rate")
    parser.add_argument('--adaptation', type=int, default=0.8, help="adaptation factor")

    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--fcw', type=int, default=50)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--trf', type=float, default=0.3, help="translation factor")
    parser.add_argument('--scalef', type=float, default=0.8, help="scaling factor")

    return parser.parse_args()

