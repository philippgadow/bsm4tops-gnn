from torch import nn
from dgl.nn import SAGEConv
import torch.nn.functional as F


class SAGENet(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, node_features, edge_features=None):
        h = self.conv1(graph, node_features, edge_features)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
