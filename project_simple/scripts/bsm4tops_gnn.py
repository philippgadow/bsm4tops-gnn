import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from os.path import join, exists
from os import makedirs
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils import visualizeDataFrame, visualizeLossAcc, visualizeROC
from utils import BSM4topsDataset, SAGENet


def getArgumentParser():
    """Get argument parser to provide command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('inputFile', help='Path to ROOT input file with 4top signal events.')
    parser.add_argument('--plot', action='store_true', help='Create overview plots.')
    parser.add_argument('--epochs', help='Number of epochs in training', type=int, default=100)
    parser.add_argument('--hidden_features', help='Number of hidden feature representations in GNN', type=int, default=400)
    parser.add_argument('--batch_size', help='Number of graphs in minibatch.', type=int, default=100)
    parser.add_argument('--raw_node_features', action='store_true', help='Do not scale and normalise node features.')
    parser.add_argument('--edge_weights', action='store_true', help='Use edge weights.')
    return parser


def evaluate(model, graph, node_features, labels, edge_features=None):
    """Evaluate accuracy of GNN SAGEConv model.
    Sets model in evaluation mode (GNN weights cannot be altered) and
    computes number of correctly predicted nodes.

    Output:
    - tuple of (n_correct, n_total) node multiplicities

    Taken and slightly adapted from: https://docs.dgl.ai/en/0.6.x/guide/training-node.html#"""
    model.eval()
    with torch.no_grad():
        if edge_features is None:  # explicity check if None to avoid ambiguities in array truth-value
            logits = model(graph, node_features)
        else:
            logits = model(graph, node_features, edge_features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0, len(labels) * 1.0


def runGNNClassifier(args):
    """Run graph-neural-network-based classification for graph nodes based on SAGEConv approach.
    - Tutorial which served as inspiration: https://docs.dgl.ai/en/0.6.x/guide/training-node.html#
    - Paper: https://arxiv.org/pdf/1706.02216.pdf

    The data is read in from the ROOT file and parsed into a graph dataset.
    More details in utils/dataset.py

    The GNN is defined in utils/model.py

    The training and evaluation is done by splitting the dataset into an 80% training and 20% testing dataset.
    For the sake of shuffling the samples, GraphDataLoaders are used but with a batch size of 1 to avoid minibatches of graphs.
    
    Args:
        args: command line arguments provided by ArgumentParser
    """
    # create dataset
    print('Creating dataset...')
    scale_node_features = not args.raw_node_features
    dataset = BSM4topsDataset(args.inputFile, scale_node_features)

    # set up GNN and optimizer
    model = SAGENet(in_feats=dataset.dim_nfeats, hid_feats=args.hidden_features, out_feats=dataset.num_classes)
    opt = torch.optim.Adam(model.parameters())

    # set up dataloaders
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)

    torch.manual_seed(42)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=False)

    # train GNN
    losses = []
    accur = []
    print('Training GNN...')
    for epoch in tqdm(range(args.epochs)):
        n_train_corr = 0.
        n_train_total = 0.
        for batched_graph, labels in train_dataloader:
            model.train()
            # forward propagation by using all nodes
            if args.edge_weights:
                pred = model(batched_graph, batched_graph.ndata['node_features'].float(), batched_graph.edata['edge_features'].float())
            else:
                pred = model(batched_graph, batched_graph.ndata['node_features'].float())
            # compute loss
            loss = F.cross_entropy(pred, torch.flatten(labels))
            # compute validation accuracy
            if args.edge_weights:
                corr, total = evaluate(model, batched_graph, batched_graph.ndata['node_features'].float(), torch.flatten(labels), batched_graph.edata['edge_features'].float())
            else:
                corr, total = evaluate(model, batched_graph, batched_graph.ndata['node_features'].float(), torch.flatten(labels))
            n_train_corr += corr
            n_train_total += total
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()
        # collect loss + accuracies
        losses.append(float(loss))
        accur.append(n_train_corr / n_train_total)

    # plot curves accuracy and loss curves
    makedirs('plots', exist_ok=True)
    visualizeLossAcc(losses, 'Loss', join('plots', 'gnn_loss_train.png'))
    visualizeLossAcc(accur, 'Accuracy', join('plots', 'gnn_accur_train.png'))

    # evaluate GNN
    n_test_corr = 0.
    n_test_total = 0.
    preds = []
    y_test = []
    for batched_graph, labels in test_dataloader:
        model.eval()
        if args.edge_weights:
            pred = model(batched_graph, batched_graph.ndata['node_features'].float(), batched_graph.edata['edge_features'].float())
        else:
            pred = model(batched_graph, batched_graph.ndata['node_features'].float())
        _, indices = torch.max(pred, dim=1)
        preds.extend(indices.detach().numpy())
        y_test.extend(torch.flatten(labels).detach().numpy())
        if args.edge_weights:
            corr, total = evaluate(model, batched_graph, batched_graph.ndata['node_features'].float(), torch.flatten(labels), batched_graph.edata['edge_features'].float())
        else:
            corr, total = evaluate(model, batched_graph, batched_graph.ndata['node_features'].float(), torch.flatten(labels))
        n_test_corr += corr
        n_test_total += total
    print(f'Test data accuracy:', n_test_corr / n_test_total)

    # plot ROC curve
    visualizeROC(preds, y_test, join('plots', 'gnn_ROC_test.png'))


def main():
    args = getArgumentParser().parse_args()

    # visualise content of dataframe
    if args.plot:
        df = getDataFrame(args.inputFile)
        df = cleanDataFrame(df)
        df = augmentDataFrame(df)
        visualizeDataFrame(df, 'plots')

    # run GNN classifier
    runGNNClassifier(args)


if __name__ == "__main__":
    main()
