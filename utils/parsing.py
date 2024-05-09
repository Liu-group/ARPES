import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='ARPESdata')
    parser.add_argument('--save_best_model', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')#./checkpoints/./models/
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mode', type=str, default='cross_val_adv_train') # predict, adv_train, cross_val_adv_train
    parser.add_argument('--adv_on', type=str, default='exp_all', help='adversarial samples used') # exp_all, exp_2014, exp_2015
    parser.add_argument('--num_adv', type=int, default=86) # number of adversarial samples to use; exp_2014: [0, 45]; exp_2015: [0, 41]; exp_all: [0, 86]
    parser.add_argument('--few', type=bool, default=True) # this decides whether loader loops on original data or target data

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--split', type=float, nargs='+', default=[0.8, 0.1, 0.1])
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--visualize', type=bool, default=False)

    parser.add_argument("--opt_goal", type=str, default='ts', help="pick the best model based on this metric")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--early_stop_epoch", type=int, default=250, help="early stop epoch")
    parser.add_argument('--conditional', type=bool, default=False, help="CDAN")
    parser.add_argument('--entropy_conditioning', type=bool, default=False, help="entropy_conditioning")
    parser.add_argument('--adaptation', type=float, default=0., help="adaptation factor")

    parser.add_argument('--hidden_channels', type=int, default=12)
    parser.add_argument('--negative_slope', type=float, default=0.01)
    parser.add_argument('--fcw', type=int, default=50)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--in_channels', type=int, default=1)

    return parser.parse_args()

