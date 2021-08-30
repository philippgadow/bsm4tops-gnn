import uproot
import numpy as np

import torch
import dgl
from torch.utils.data import Dataset




class JetEventsDataset(Dataset):

    def __init__(self, filepath, jet_tag, entrystart=0, entrystop=None):
        '''
        Args:
            filepath   : path to the root file with (fetures + labels)
            jet_tag    : which tag to use
            entrystart : 
            entrystop  :
        '''


        self.filepath = filepath
        self.f = uproot.open(self.filepath)['Nominal']
        self.jet_tag = jet_tag

        if entrystop == None:
            entrystop =  self.f.numentries

        # precaution
        entrystart, entrystop = int(entrystart), int(entrystop)
        self.n_events = (entrystop - entrystart)

        # b'features', b'jet_truthflav', b'deltaR', b'jet_tag'
        self.brancehs2read = ['features', 'jet_truthflav', self.jet_tag, 'deltaR', 'jet_flavindex']

        data = self.f.arrays(self.brancehs2read, entrystart=entrystart, entrystop=entrystop)

        # combine the datasets
        self.features      = data[b'features']
        self.jet_truthflav = data[b'jet_truthflav']
        self.jet_tag       = data[bytes(self.jet_tag, 'utf-8')]
        self.deltaR        = data[b'deltaR']
        self.jet_flavindex = data[b'jet_flavindex']


        # src and dst dict
        self.edge_dict = {}
        for n in range(50):
            ind = np.arange(n**2)%(n+1) != 0 
            src = np.repeat(np.arange(n), n)[ind]
            dst = np.tile(np.arange(n), n)[ind]
            self.edge_dict[n] = [src, dst]


    def __len__(self):
        return self.n_events


    def __getitem__(self, idx):

        x = torch.FloatTensor(self.features[idx]).reshape(-1, 5)

        n,_ = x.shape
        g = dgl.DGLGraph()
        g.add_nodes(n)
        g.ndata['node_features'] = x
        g.ndata['flav_indices'] = torch.LongTensor(self.jet_flavindex[idx])

        src, dst = self.edge_dict[n]
        g.add_edges(src, dst)
        dr = self.deltaR[idx]
        g.edata['dR'] = torch.FloatTensor(dr)

        flav = torch.FloatTensor(self.jet_truthflav[idx])
        target = torch.FloatTensor(self.jet_tag[idx])

        return g, flav, target



def collate(samples):

    graphs  = [x[0] for x in samples]
    flavs   = [x[1] for x in samples]
    targets = [x[2] for x in samples]
    
    batched_graph = dgl.batch(graphs,node_attrs=['node_features', 'flav_indices'],edge_attrs=['dR'])
    flavs = torch.cat(flavs)
    targets = torch.cat(targets)
    
    return batched_graph, flavs.unsqueeze(1).float(), targets.unsqueeze(1).float() 

