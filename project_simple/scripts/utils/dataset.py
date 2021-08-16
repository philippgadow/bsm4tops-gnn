import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from os.path import join


class BSM4topsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BSM4topsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ['bsm4tops_dataset.pkl']
    @property
    def processed_file_names(self):
        return ['bsm4tops.dataset']
    def download(self):
        pass
    def process(self):
        data_list = []
        # load top quark events
        df = pd.read_pickle(join("data", "raw", "bsm4tops_dataset.pkl"))

        # loop over events and create dataset
        for event, new_df in tqdm(df.groupby(level=0)):
            new_df = new_df.sample(frac=1).reset_index(drop=True)
            edge_index = torch.tensor(
                [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                 [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]],
                dtype=torch.long)
            x = torch.tensor([[new_df['Particle.PT'].to_numpy()[0], new_df['Particle.Eta'].to_numpy()[0], new_df['Particle.Phi'].to_numpy()[0], new_df['Particle.M'].to_numpy()[0]],
                              [new_df['Particle.PT'].to_numpy()[1], new_df['Particle.Eta'].to_numpy()[1], new_df['Particle.Phi'].to_numpy()[1], new_df['Particle.M'].to_numpy()[0]],
                              [new_df['Particle.PT'].to_numpy()[2], new_df['Particle.Eta'].to_numpy()[2], new_df['Particle.Phi'].to_numpy()[2], new_df['Particle.M'].to_numpy()[0]],
                              [new_df['Particle.PT'].to_numpy()[3], new_df['Particle.Eta'].to_numpy()[3], new_df['Particle.Phi'].to_numpy()[3], new_df['Particle.M'].to_numpy()[0]]],
                                  dtype=torch.float)
            y = torch.LongTensor(new_df['resonance'].to_numpy())
            data = Data(x=x, y=y, edge_index=edge_index)
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
