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

    print(f'Dataset: {dataset}:')
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
    visualizeGraph(data, join('plots', 'plot_graph_example.png'))

    # build GNN
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.nll_loss

    # train model
    model.train()

    def train(model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    for epoch in range(10):
        for i in range(len(dataset))[0:int(0.8*len(dataset))]:
            data = dataset[i]
            loss = train(model, data, optimizer, criterion)
            print(f'Epoch: {epoch:03d}, Graph: {i:04d}, Loss: {loss:.4f}')


    # evaluate model
    model.eval()

    # TODO: re-write evaluation function such that it provides a summary not only for the last graph
    #       which happens to be the last element of "dataset"

    def test(model, data):
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        # test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return int(test_correct.sum()), int(data.test_mask.sum())

    a_tot = 0
    b_tot = 0
    for i in range(len(dataset))[int(0.8*len(dataset)): -1]: 
        a, b = test(model, data)
        a_tot += a
        b_tot += b
    test_acc = a_tot / b_tot
    print(f'Test Accuracy: {test_acc:.4f}')


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
