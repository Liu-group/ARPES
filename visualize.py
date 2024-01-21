from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from dataset import ARPESDataset
from torch.utils.data import DataLoader
import torch
from utils.utils import load_checkpoint
from utils.parsing import parse_args
import time

def plot_tsne(X, y, save_path, classification):
    X_embedded = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0).fit_transform(X)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', s=5)
    # produce a legend with the unique colors from the scatter
    if classification:
        labels = ['class 0', 'class 1', 'class 2']
    else:
        labels=['sim', 'exp_2014', 'exp_2015']
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    ax.set_title(save_path.split('/')[-1].split('.')[0])
    fig.savefig(save_path, bbox_inches='tight')
    print("Saved to ", save_path)
    plt.show()
    plt.close(fig)

def visualize(args, model):
    """ visualize the data """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = args.num_classes
    # load sim data
    x_sim = np.load('./ARPESdata/data_sim_processed_nobg.npy')
    y_sim = np.load('./ARPESdata/data_sim_sc_values.npy')
    class_sim = [[0]*len(y_sim)]
    # load exp data
    x_exp_2014 = np.load('./ARPESdata/data_exp_2014_processed_nobg.npy')
    y_exp_2014 = np.load('./ARPESdata/data_exp_2014_sc_values.npy')
    class_2014 = [[1]*len(y_exp_2014)]

    x_exp_2015 = np.load('./ARPESdata/data_exp_2015_processed_nobg.npy')
    y_exp_2015 = np.load('./ARPESdata/data_exp_2015_sc_values.npy')
    class_2015 = [[2]*len(y_exp_2015)]

    if num_classes==2:
        y_sim[y_sim==1]=0
        y_sim[y_sim==2]=1
        y_exp_2014[y_exp_2014==1]=0
        y_exp_2014[y_exp_2014==2]=1
        y_exp_2015[y_exp_2015==1]=0
        y_exp_2015[y_exp_2015==2]=1

    print("number of sim samples: ", len(y_sim))
    print("number of exp_2014 samples: ", len(y_exp_2014))
    print("number of exp_2015 samples: ", len(y_exp_2015))
    domain_label_all = np.concatenate((np.array(class_sim).reshape(-1), np.array(class_2014).reshape(-1), np.array(class_2015).reshape(-1)), axis=0)
    class_label_all = np.concatenate((y_sim, y_exp_2014, y_exp_2015), axis=0)
    x_all = np.concatenate((x_sim, x_exp_2014, x_exp_2015), axis=0)
    data_all = ARPESDataset(x_all, class_label_all)
    loader_all = DataLoader(data_all, batch_size=args.batch_size, shuffle=False)

    model.eval().to(device)

    inputs = []
    def hook(module, input, output):
        inputs.append(input)

    model.class_classifier.register_forward_hook(hook)

    with torch.no_grad():
        for data in loader_all:
            X, _, _ = data
            X = X.unsqueeze(1).double().to(device)
            out, _ = model(X, alpha=0)


    inputs = [i for sub in inputs for i in sub]
    inputs = torch.cat(inputs)
    inputs = inputs.cpu().numpy()
    print("Number of samples: ", inputs.shape[0])
    print("Number of features: ", inputs.shape[1])
    from datetime import datetime
    time = datetime.now().strftime('%m-%d-%H:%M')
    plot_tsne(inputs, domain_label_all, f'visualization/{time}_{args.seed}_domian_tsne.png', classification=False)
    plot_tsne(inputs, class_label_all, f'visualization/{time}_{args.seed}_class_tsne.png', classification=True)

    del model    
    
if __name__ == '__main__':
    args = parse_args()
    print(args)
    model = load_checkpoint(args)
    visualize(args, model) 
