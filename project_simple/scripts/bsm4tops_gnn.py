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
    dataset = BSM4topsDataset('data/')
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
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    # visualize graph
    visualizeGraph(data, join('plots', 'plot_graph_example.png'))

    # # build GNN - shamelessly stolen and slightly modified from Javier Duarte's course
    # # https://github.com/jmduarte/capstone-particle-physics-domain/blob/master/weeks/08-extending.ipynb
    # import torch.nn as nn
    # import torch.nn.functional as F
    # import torch_geometric.transforms as T
    # from torch_geometric.nn import EdgeConv, global_mean_pool
    # from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
    # from torch_scatter import scatter_mean
    # from torch_geometric.nn import MetaLayer

    # inputs = 4
    # hidden = 12
    # outputs = 1

    # class EdgeBlock(torch.nn.Module):
    #     def __init__(self):
    #         super(EdgeBlock, self).__init__()
    #         self.edge_mlp = Seq(Lin(inputs*2, hidden), 
    #                             BatchNorm1d(hidden),
    #                             ReLU(),
    #                             Lin(hidden, hidden))

    #     def forward(self, src, dest, edge_attr, u, batch):
    #         out = torch.cat([src, dest], 1)
    #         return self.edge_mlp(out)
    # class NodeBlock(torch.nn.Module):
    #     def __init__(self):
    #         super(NodeBlock, self).__init__()
    #         self.node_mlp_1 = Seq(Lin(inputs+hidden, hidden), 
    #                               BatchNorm1d(hidden),
    #                               ReLU(), 
    #                               Lin(hidden, hidden))
    #         self.node_mlp_2 = Seq(Lin(inputs+hidden, hidden), 
    #                               BatchNorm1d(hidden),
    #                               ReLU(), 
    #                               Lin(hidden, hidden))

    #     def forward(self, x, edge_index, edge_attr, u, batch):
    #         row, col = edge_index
    #         out = torch.cat([x[row], edge_attr], dim=1)
    #         out = self.node_mlp_1(out)
    #         out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
    #         out = torch.cat([x, out], dim=1)
    #         return self.node_mlp_2(out)
    # class GlobalBlock(torch.nn.Module):
    #     def __init__(self):
    #         super(GlobalBlock, self).__init__()
    #         self.global_mlp = Seq(Lin(hidden, hidden),                               
    #                               BatchNorm1d(hidden),
    #                               ReLU(), 
    #                               Lin(hidden, outputs))

    #     def forward(self, x, edge_index, edge_attr, u, batch):
    #         out = scatter_mean(x, batch, dim=0)
    #         return self.global_mlp(out)
    # class InteractionNetwork(torch.nn.Module):
    #     def __init__(self):
    #         super(InteractionNetwork, self).__init__()
    #         self.interactionnetwork = MetaLayer(EdgeBlock(), NodeBlock(), GlobalBlock())
    #         self.bn = BatchNorm1d(inputs)
            
    #     def forward(self, x, edge_index, batch):
            
    #         x = self.bn(x)
    #         x, edge_attr, u = self.interactionnetwork(x, edge_index, None, None, batch)
    #         return u

    # model = InteractionNetwork()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # @torch.no_grad()
    # def test(model,loader,total,batch_size,leave=False):
    #     model.eval()
    #     xentropy = nn.CrossEntropyLoss(reduction='mean')
    #     sum_loss = 0.
    #     t = tqdm(enumerate(loader),total=total/batch_size,leave=leave)
    #     for i,data in t:
    #         data = data.to(device)
    #         y = torch.argmax(data.y,dim=1)
    #         batch_output = model(data.x, data.edge_index, data.batch)
    #         batch_loss_item = xentropy(batch_output, y).item()
    #         sum_loss += batch_loss_item
    #         t.set_description("loss = %.5f" % (batch_loss_item))
    #         t.refresh() # to show immediately the update
    #     return sum_loss/(i+1)

    # def train(model, optimizer, loader, total, batch_size,leave=False):
    #     model.train()
    #     xentropy = nn.CrossEntropyLoss(reduction='mean')
    #     sum_loss = 0.
    #     t = tqdm(enumerate(loader),total=total/batch_size,leave=leave)
    #     for i, data in t:
    #         data = data.to(device)
    #         y = torch.argmax(data.y,dim=1)
    #         optimizer.zero_grad()
    #         batch_output = model(data.x, data.edge_index, data.batch)
    #         batch_loss = xentropy(batch_output, y)
    #         batch_loss.backward()
    #         batch_loss_item = batch_loss.item()
    #         t.set_description("loss = %.5f" % batch_loss_item)
    #         t.refresh() # to show immediately the update
    #         sum_loss += batch_loss_item
    #         optimizer.step()
    #     return sum_loss/(i+1)

    # from torch_geometric.data import Data, DataListLoader, Batch
    # from torch.utils.data import random_split

    # def collate(items):
    #     return items
    #     l = sum(items, [])
    #     return Batch.from_data_list(l)


    # torch.manual_seed(0)
    # valid_frac = 0.10
    # full_length = len(dataset)
    # valid_num = int(valid_frac*full_length)
    # batch_size = 32

    # train_dataset, valid_dataset = random_split(dataset, [full_length-valid_num,valid_num])

    # train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    # train_loader.collate_fn = collate
    # valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    # valid_loader.collate_fn = collate
    # # test_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    # # test_loader.collate_fn = collate

    # train_samples = len(train_dataset)
    # valid_samples = len(valid_dataset)
    # # test_samples = len(valid_dataset)

    # import os.path as osp
    # n_epochs = 10
    # stale_epochs = 0
    # best_valid_loss = 99999
    # patience = 5
    # t = tqdm(range(0, n_epochs))

    # for epoch in t:
    #     # loss = train(model, optimizer, train_loader, train_samples, batch_size,leave=bool(epoch==n_epochs-1))
    #     loss = train(model, optimizer, train_dataset, train_samples, batch_size,leave=bool(epoch==n_epochs-1))
    #     # valid_loss = test(model, valid_loader, valid_samples, batch_size,leave=bool(epoch==n_epochs-1))
    #     valid_loss = test(model, valid_dataset, valid_samples, batch_size,leave=bool(epoch==n_epochs-1))
    #     print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
    #     print('           Validation Loss: {:.4f}'.format(valid_loss))

    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         modpath = osp.join('interactionnetwork_best.pth')
    #         print('New best model saved to:',modpath)
    #         torch.save(model.state_dict(),modpath)
    #         stale_epochs = 0
    #     else:
    #         print('Stale epoch')
    #         stale_epochs += 1
    #     if stale_epochs >= patience:
    #         print('Early stopping after %i stale epochs'%patience)
    #         break


    # model.eval()
    # t = tqdm(enumerate(test_loader),total=test_samples/batch_size)
    # y_test = []
    # y_predict = []
    # for i,data in t:
    #     data = data.to(device)    
    #     batch_output = model(data.x, data.edge_index, data.batch)    
    #     y_predict.append(batch_output.detach().cpu().numpy())
    #     y_test.append(data.y.cpu().numpy())
    # y_test = np.concatenate(y_test)
    # y_predict = np.concatenate(y_predict)


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
