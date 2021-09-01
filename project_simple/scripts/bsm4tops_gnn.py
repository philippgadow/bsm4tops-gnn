import numpy as np
import pandas as pd

from argparse import ArgumentParser
from os import makedirs
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn, optim

from utils import getDataFrame, cleanDataFrame, augmentDataFrame
from utils import visualizeDataFrame, visualizeGraph, visualizeLossAcc, visualizeROC
from utils import BSM4topsDataset

def getArgumentParser():
    parser = ArgumentParser()
    parser.add_argument('inputFile')
    parser.add_argument('--gnn', action='store_true', help='Run graph neural network and evaluate its performance.')
    parser.add_argument('--plot', action='store_true', help='Create overview plots.')
    return parser


def get_device():
    """Determine supported device: use GPU if avaliable, otherwise use CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device


def runGNNClassifier(df):
    """Run graph-neural-network-based classification.
    
    Args:
        X_train (pandas dataframe): training data, contains features for training
        X_test (pandas dataframe): test data, contains features for evaluating performance
        y_train (pandas series): training targets, contains targets for training
        y_test (pandas series): test targets, contains true targets to evaluate performance of prediction
    """

    device = get_device()

    # save training data to disk
    makedirs('data/raw/', exist_ok=True)
    df.to_pickle(join("data", "raw", "bsm4tops_dataset.pkl"))

    # create dataset and save processed data also to disk
    dataset = BSM4topsDataset('data/', 'bsm4tops')
    dataset = dataset.shuffle()

    print(f"Dataset: {dataset}:")
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Get the first graph object in training dataset.
    data = dataset[0]

    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of node features: {data.num_node_features}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    # visualize graph
    #visualizeGraph(data, join('plots', 'plot_graph_example.png'))

    # build GNN
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_max_pool
    
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            self.conv2 = GCNConv(16, 16)
            self.lin = nn.Linear(16,6)
        def forward(self, x, edge_index, batch):

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)

            x = global_max_pool(x, batch)

            x = F.dropout(x, training=self.training)
            x = self.lin(x)
            return x

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.nll_loss
    from torch_geometric.data import DataLoader

    train_dataset = dataset[:18000]
    test_dataset = dataset[18000:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False) 

    # train model

    def train():
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    for epoch in range(1, 20):
        train()
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')

def main():
    # get command line arguments
    args = getArgumentParser().parse_args()

    # parse input file into a pandas dataframe
    df = getDataFrame(args.inputFile)
    df = cleanDataFrame(df)

    # augment dataframe with additional variables
    df = augmentDataFrame(df)

    # shuffle data frame
    df = shuffle(df)

    # visualise content of dataframe
    if args.plot:
        visualizeDataFrame(df, 'plots')

    # run the GNN classifier
    target_names = ['resonance']
    feature_names = ['Particle.PT', 'Particle.Eta', 'Particle.Phi', 'Particle.M']
    if args.gnn:
        runGNNClassifier(df)


if __name__ == '__main__':
    main()
