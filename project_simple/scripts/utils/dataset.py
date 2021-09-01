import torch
import dgl
import numpy as np
from dgl.data import DGLDataset
from tqdm import tqdm
from copy import deepcopy
from utils.io import getDataFrame, cleanDataFrame, augmentDataFrame
from utils.functions import compute_dR


class BSM4topsDataset(DGLDataset):
    def __init__(self, inputFile=None, normalise_features=True):
        self.inputFile = inputFile
        self.normalise_features = normalise_features
        super().__init__(name='bsm4tops_dataset')

    def process(self):
        self.graphs = []
        self.labels = []

        self.df = getDataFrame(self.inputFile)
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
            node_features = new_df[['Particle.PT', 'Particle.Eta', 'Particle.Phi', 'Particle.M']].astype(np.float32).to_numpy()
            edge_features = compute_dR(new_df, node_features, edges_src, edges_dst)
            # take inverse delta_R as edge features to weight nodes close-by with higher importance
            edge_features = np.reciprocal(edge_features)

            # normalise features in preprocessing
            if self.normalise_features:
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                node_features = sc.fit_transform(node_features)

            node_features = torch.from_numpy(node_features)
            edge_features = torch.from_numpy(edge_features)
            node_labels = torch.from_numpy(deepcopy(new_df['resonance'].astype(np.int64).to_numpy()))

            # construct DGL graph
            g = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            g.ndata['node_features'] = node_features
            g.ndata['label'] = node_labels
            g.edata['edge_features'] = edge_features

            self.graphs.append(g)
            self.labels.append(node_labels)

        # number of classes in dataset
        self.num_classes = len(torch.unique(torch.LongTensor(node_labels)))
        # dimension of node features
        self.dim_nfeats = node_features.shape[1]

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
