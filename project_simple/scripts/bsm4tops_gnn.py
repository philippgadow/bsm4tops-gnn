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
    parser = ArgumentParser()
    parser.add_argument('inputFile', help='Path to ROOT input file with 4top signal events.')
    parser.add_argument('--epochs', help='Number of epochs in training', type=int, default=20)
    parser.add_argument('--hidden_features', help='Number of hidden feature representations in GNN', type=int, default=100)
    return parser


def evaluate(model, graph, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0, len(labels)


def runGNNClassifier(args):
    # create dataset
    print('Creating dataset...')
    dataset = BSM4topsDataset(args.inputFile)

    # set up GNN and optimizer
    model = SAGENet(in_feats=dataset.dim_nfeats, hid_feats=args.hidden_features, out_feats=dataset.num_classes)
    opt = torch.optim.Adam(model.parameters())

    # set up dataloaders
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=1, drop_last=False)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=1, drop_last=False)

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
            pred = model(batched_graph, batched_graph.ndata['node_features'].float())
            # compute loss
            loss = F.cross_entropy(pred, torch.flatten(labels))
            # compute validation accuracy
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
        pred = model(batched_graph, batched_graph.ndata['node_features'].float())
        _, indices = torch.max(pred, dim=1)
        preds.extend(indices.detach().numpy())
        y_test.extend(torch.flatten(labels).detach().numpy())
        corr, total = evaluate(model, batched_graph, batched_graph.ndata['node_features'].float(), torch.flatten(labels))
        n_test_corr += corr
        n_test_total += total
    print(f'Test data accuracy:', n_test_corr / n_test_total)

    # plot ROC curve
    visualizeROC(preds, y_test, join('plots', 'gnn_ROC_test.png'))


def main():
    args = getArgumentParser().parse_args()
    runGNNClassifier(args)


if __name__ == "__main__":
    main()
