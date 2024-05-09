from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from dataset import ARPESDataset
from torch.utils.data import DataLoader
import torch
from utils.utils import load_checkpoint, normalize_transform
from utils.parsing import parse_args
import time
plt.rcParams['pdf.fonttype'] = 'truetype'
#plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams["font.size"] = 8
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams.update({
    "pdf.use14corefonts": True
})

def plot_tsne(X, y, save_path, classification):
    X_embedded = TSNE(n_components=2, perplexity=100, n_iter=5000, random_state=123).fit_transform(X)
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    # produce a legend with the unique colors from the scatter
    if classification=='SC':
        labels = ['non-SC', 'fluc-SC', 'SC']
        cmap = 'Accent'
    elif classification == 'domain':
        labels=['simulated', 'BSCCO OD58', 'BSCCO OD50']
        cmap = ListedColormap(['orange', 'green', 'cyan'])#'tab10'
    else:
        labels = ['Simulated non-SC', 'Simulated SC', 'Experimental non-SC', 'Experimental SC']
        cmap = ListedColormap([ '#6fb39b','#dc8b94', '#377444','#813e3e' ])
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=cmap, s=1)
    #plt.legend(handles=scatter.legend_elements()[0], labels=labels, loc='upper left', fontsize=3)
    #ax.set_title(save_path.split('/')[-1].split('.')[0])
    # remove the ticks
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    
    fig.savefig(save_path+'.eps', format = 'eps', bbox_inches='tight', dpi=300)
    fig.savefig(save_path+'.png', format = 'png', bbox_inches='tight', dpi=300)
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
    
    dataset_sim = ARPESDataset(x_sim, transform=normalize_transform('sim'))
    dataset_2014 = ARPESDataset(x_exp_2014, transform=normalize_transform('exp_2014'))
    dataset_2015 = ARPESDataset(x_exp_2015, transform=normalize_transform('exp_2015'))
    data_all = torch.utils.data.ConcatDataset([dataset_sim, dataset_2014, dataset_2015])
    loader_all = DataLoader(data_all, batch_size=args.batch_size, shuffle=False)
    model.eval().to(device)

    inputs = []
    def hook(module, input, output):
        inputs.append(input)

    #model.class_classifier[6].register_forward_hook(hook)
    model.class_classifier.register_forward_hook(hook)

    with torch.no_grad():
        for data in loader_all:
            X, _ = data
            X = X.unsqueeze(1).double().to(device)
            out = model(X, alpha=0)

    inputs = [i for sub in inputs for i in sub]
    inputs = torch.cat(inputs)
    inputs = inputs.cpu().numpy()
    print("Number of samples: ", inputs.shape[0])
    print("Number of features: ", inputs.shape[1])
    from datetime import datetime
    time = datetime.now().strftime('%m-%d-%H:%M')
    #plot_tsne(inputs, domain_label_all, f'visualization/{time}_{args.seed}_domain_tsne', classification='domain')
    #plot_tsne(inputs, class_label_all, f'visualization/{time}_{args.seed}_class_tsne', classification='SC')
    domain_labels = np.concatenate((np.array(class_sim).reshape(-1),  np.array([[1]*(len(y_exp_2014)+len(y_exp_2015))]).reshape(-1)), axis=0)
    composite_labels = 2 * domain_labels + class_label_all
    #print(class_label_all)
    plot_tsne(inputs, composite_labels, f'visualization/{time}_{args.seed}_composite', classification='comb')
    del model    
    
if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    args = parse_args()
    print(args)
    model = load_checkpoint(args)
    visualize(args, model) 
