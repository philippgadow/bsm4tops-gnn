import os
import uproot
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib
matplotlib.rcParams.update({'font.size': 10})

import mplhep as hep
plt.style.use([hep.style.ATLAS])


def plot_hist(filepath, path2save, title):
    
    f = uproot.open(filepath)    
    hists = f.keys()

    fig = plt.figure(figsize=(3*6, 3*6), dpi=150)
    gs = gridspec.GridSpec(5, 3, hspace=0.6) 
    plt.suptitle(title, y=0.92)

    for var_num, var in enumerate(['pt', 'eta', 'phi', 'mass', 'mu']):
        for flav_num, flav in enumerate(['b', 'c', 'l']):

            # single jet events
            edges_single     = f[var + '_hist_single_jet_' + flav].edges
            values_single    = f[var + '_hist_single_jet_' + flav].values
            variances_single = f[var + '_hist_single_jet_' + flav].variances

            # atleast two jets events
            edges_atleast_two     = f[var + '_hist_atleast_two_jets_' + flav].edges
            values_atleast_two    = f[var + '_hist_atleast_two_jets_' + flav].values
            variances_atleast_two = f[var + '_hist_atleast_two_jets_' + flav].variances

            x = 0.5 * (edges_single[:-1] + edges_single[1:])
            bin_width = edges_single[1] - edges_single[0]

            gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, gs[var_num, flav_num], hspace=0)
            ax1 = fig.add_subplot(gs_inner[0])
            ax2 = fig.add_subplot(gs_inner[1])

            ax1.bar(x, values_single, width=bin_width)
            ax2.bar(x, values_atleast_two, width=bin_width)

            ax1.set_xlim([edges_single[0], edges_single[-1]+bin_width])
            ax2.set_xlim([edges_single[0], edges_single[-1]+bin_width])

            var_title = var
            if var == 'mass':
                var_title = 'mJ'

            ax1.set_title('{} ({})'.format(var_title, flav), y=1.07)

    plt.savefig(path2save, bbox_inches='tight')


filepath  = '/eos/user/e/eloiacon/SWAN_projects/GNN/ATLAS_GNN/latest_samples/ttbar_nonallhad.root'
path2save = '/eos/user/e/eloiacon/SWAN_projects/GNN/ATLAS_GNN/latest_samples/tt_leading_jet_dist.png'
title     = 'single jet events vs atleast two jets events'

plot_hist(filepath, path2save, title)
