import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from copy import deepcopy
from tqdm import tqdm
from dgl.nn import GraphConv

from utils import getDataFrame, cleanDataFrame, augmentDataFrame
from utils import visualizeDataFrame, visualizeGraph, visualizeLossAcc, visualizeROC
from utils import BSM4topsDataset


# create own dataset
from dgl.data import DGLDataset
class BSM4topsDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='bsm4tops_dataset')

    def process(self):
        self.graphs = []
        self.labels = []

        self.df = getDataFrame("../data/simple/unweighted_events.root")
        self.df = cleanDataFrame(self.df)
        self.df = augmentDataFrame(self.df)

        # loop over events and create dataset
        for event, new_df in tqdm(self.df.groupby(level=0)):
            new_df = new_df.sample(frac=1).reset_index(drop=True)

            # define graph (fully connected graph of four nodes)
            edges_src = torch.LongTensor([0, 0, 0, 1, 1, 2])
            edges_dst = torch.LongTensor([1, 2, 3, 0, 2, 3])
            num_nodes = 4

            # graph consists of:
            # - four nodes (top quarks at parton level) with features
            #   - pt, eta, phi, mass
            # - six edges (links between top quarks t1 and t2) with features
            #   - dR(t1, t2)
            node_features = torch.from_numpy(new_df[['Particle.PT', 'Particle.Eta', 'Particle.Phi', 'Particle.M']].astype(np.float32).to_numpy())
            node_labels = torch.from_numpy(deepcopy(new_df['resonance'].astype(np.int64).to_numpy()))
            def compute_dR(node_features, edges_src, edges_dst):
                import vector
                result = []
                v_array = vector.arr({
                    "pt": new_df['Particle.PT'],
                    "phi": new_df['Particle.Eta'],
                    "eta": new_df['Particle.Phi'],
                    "M": new_df['Particle.M'],
                })
                for i, j in zip(edges_src, edges_dst):
                    result.append(v_array[i].deltaR(v_array[j]))
                return np.array(result)
            edge_features = torch.from_numpy(compute_dR(node_features, edges_src, edges_dst))

            g = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            g.ndata['node_features'] = node_features
            g.ndata['label'] = node_labels
            g.edata['edge_features'] = edge_features

            # define masks
            g.ndata['train_mask'] = torch.ones(num_nodes, dtype=torch.bool)
            g.ndata['val_mask'] = torch.ones(num_nodes, dtype=torch.bool)
            g.ndata['test_mask'] = torch.ones(num_nodes, dtype=torch.bool)

            # define graph label
            label = int(node_labels[0]) * 1000 + \
                    int(node_labels[1]) * 100 + \
                    int(node_labels[2]) * 10 + \
                    int(node_labels[3]) * 1

            self.graphs.append(g)
            self.labels.append(label_dict[f'{label:04}'])

        # number of classes in dataset (indicates position of resonance top quarks)
        self.num_classes = len(torch.unique(torch.LongTensor(self.labels)))
        # dimension of node features
        self.dim_nfeats = node_features.shape[1]


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

# create GNN
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

# dictionary to encode graph types
label_dict = {
    '1100': 0,
    '1010': 1,
    '1001': 2,
    '0110': 3,
    '0101': 4,
    '0011': 5
}

# run code
# create dataset
dataset = BSM4topsDataset()

# set up dataloader
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=32, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=32, drop_last=False)


# set up model
model = GCN(dataset.dim_nfeats, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in tqdm(range(100)):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['node_features'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['node_features'].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)
