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
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, MetaLayer
from torch_geometric.utils import accuracy
from torch_geometric.transforms import NormalizeFeatures


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
    dataset = BSM4topsDataset('data/', 'bsm4tops', transform=NormalizeFeatures())
    dataset = dataset.shuffle()

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # get the first graph object in training dataset.
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
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 12)
            self.conv2 = GCNConv(12, dataset.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)


    class EdgeBlock(torch.nn.Module):
        def __init__(self):
            super(EdgeBlock, self).__init__()
            self.hidden = 12
            outputs = dataset.num_classes
            self.edge_mlp = nn.Sequential(
                nn.Linear(dataset.num_node_features*2, self.hidden), 
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden))
        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest], 1)
            return self.edge_mlp(out)

    class NodeBlock(torch.nn.Module):
        def __init__(self):
            super(NodeBlock, self).__init__()
            self.hidden = 12
            self.node_mlp_1 = nn.Sequential(
                nn.Linear(dataset.num_node_features+self.hidden, self.hidden), 
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(), 
                nn.Linear(self.hidden, self.hidden))
            self.node_mlp_2 = nn.Sequential(
                nn.Linear(dataset.num_node_features+self.hidden, self.hidden), 
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(), 
                nn.Linear(self.hidden, self.hidden))
        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out], dim=1)
            return self.node_mlp_2(out)
    
    class GlobalBlock(torch.nn.Module):
        def __init__(self):
            super(GlobalBlock, self).__init__()
            self.hidden = 12
            self.global_mlp = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),                               
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(), 
                nn.Linear(self.hidden, dataset.num_classes))
        def forward(self, x, edge_index, edge_attr, u, batch):
            out = scatter_mean(x, batch, dim=0)
            return self.global_mlp(out)

    class InteractionNetwork(torch.nn.Module):
        def __init__(self):
            super(InteractionNetwork, self).__init__()
            self.interactionnetwork = MetaLayer(EdgeBlock(), NodeBlock(), GlobalBlock())
            self.bn = nn.BatchNorm1d(dataset.num_node_features)
            
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.bn(x)
            x, edge_attr, u = self.interactionnetwork(x, edge_index, None, None, batch)
            return u
        
    # set up model
    # model = Net()
    model = InteractionNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = F.cross_entropy

    # set up data loader
    from torch_geometric.data import DataLoader
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # train model (set model in training mode - modify weights)
    model.train()
    # def train(model, data, optimizer, criterion):
    #     optimizer.zero_grad()
    #     out = model(data)
    #     loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    #     loss.backward()  # Derive gradients.
    #     optimizer.step()  # Update parameters based on gradients.
    #     return loss
    # for epoch in range(10):
    #     for data in loader:
    #         loss = train(model, data, optimizer, criterion)
    #         _, pred = model(data).max(dim=1)
    #         acc = accuracy(pred, data.y)
    #         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')


    def train(model, optimizer, loader, total, batch_size, leave=False):
        xentropy = nn.CrossEntropyLoss(reduction='mean')
        sum_loss = 0.
        t = tqdm(enumerate(loader), total=total / batch_size, leave=leave)
        for i, data in t:
            optimizer.zero_grad()
            batch_output = model(data)
            print(batch_output)
            batch_loss = xentropy(batch_output, data.y)
            batch_loss.backward()
            batch_loss_item = batch_loss.item()
            t.set_description("loss = %.5f" % batch_loss_item)
            t.refresh() # to show immediately the update
            sum_loss += batch_loss_item
            optimizer.step()
        return sum_loss / (i + 1)

    n_epochs = 10
    train_samples = len(dataset)
    t = tqdm(range(0, n_epochs))

    for epoch in t:
        loss = train(model, optimizer, loader, train_samples, 128, leave=bool(epoch==n_epochs-1))
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))


    # # evaluate model (set model into evaluation mode / fix weights)
    # model.eval()
    # def test(model, data):
    #     model.eval()
    #     out = model(data)
    #     pred = out.argmax(dim=1)  # Use the class with highest probability.
    #     test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    #     test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    #     return test_acc

    # test_acc = test(model, data)
    # print(f'Test Accuracy: {test_acc:.4f}')


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
