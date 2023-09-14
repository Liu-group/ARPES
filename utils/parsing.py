import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='ARPESdata')
    parser.add_argument('--save_best_model', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='./models/model_3_class_new.pt')
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--mode', type=str, default='train') # train, predict
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dataset_name', type=str, default='exp_2014')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--split', type=list, default=[0.8, 0.1, 0.1])

    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    

    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--fcw', type=int, default=50)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--trf', type=float, default=0.3, help="translation factor")
    parser.add_argument('--scalef', type=float, default=0.6, help="scaling factor")

    return parser.parse_args([])

