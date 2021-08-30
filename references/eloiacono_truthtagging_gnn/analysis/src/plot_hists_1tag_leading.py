import os
import uproot
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

import matplotlib
matplotlib.rcParams.update({'font.size': 10})

import mplhep as hep
plt.style.use([hep.style.ATLAS])

import warnings

def load_hist(filepath, var):
    
    f = uproot.open(filepath)    
    hists = f.keys()
    
    data = {}
    
    for flav in ['b', 'c', 'l']:
        tmp_data = [f[var + '_hist_direct_' + flav].edges]

        # direct tag
        tmp_data.append(f[var + '_hist_direct_' + flav].values)
        tmp_data.append(np.sqrt(f[var + '_hist_direct_' + flav].variances))

        # custom efficiency
        tmp_data.append(f[var + '_hist_custom_' + flav].values)
        tmp_data.append(np.sqrt(f[var + '_hist_custom_' + flav].variances))

        # NN efficiency
        tmp_data.append(f[var + '_hist_NN_' + flav].values)
        tmp_data.append(np.sqrt(f[var + '_hist_NN_' + flav].variances))

        data[flav] = tmp_data
        
    return data



def plot_hist(data, flav, fig, gs, xlabel):

    dir_x, dir_h, dir_err, custom_h, custom_err, NN_h, NN_err = data[flav]
    
    x = 0.5*(dir_x[:-1] + dir_x[1:])
    bin_width = x[1]-x[0]

    gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, gs, hspace=0)
    ax1 = fig.add_subplot(gs_inner[0])
    ax2 = fig.add_subplot(gs_inner[1])

    # hist plot
    ax1.bar(x,dir_h,width=bin_width,fc='cornflowerblue',alpha=0.5,label='Direct Tag')
    ax1.errorbar(x,dir_h,dir_err,linestyle='none',c='k',marker='o',markersize=2)

    ax1.errorbar(x,custom_h,custom_err,c='lightslategray',label='custom map',linewidth=2)

    ax1.errorbar(x,NN_h,NN_err,c='r',label='NN',linewidth=2.6)
    
    ax1.legend()
    ax1.set_ylabel('Events')
    hep.atlas.label(ax=ax1, llabel="Simulation Internal", rlabel=r'$\sqrt{s}$ = 13 TeV', fontsize=11)

    flav_dict = {
        'b': 'bottom', 'c': 'charm', 'l': 'light',
        'bb': 'bottom bottom', 'cc': 'charm charm', 'll': 'light light',
        'bc': 'bottom charm', 'bl': 'bottom light', 'cl': 'charm light'
    }
    ax1.set_title(flav_dict[flav], y=1.1)
    ax1.set_xlim(x[0], x[-1])
    
    # ratio plot
    ax2.plot([x[0],x[-1]],[0,0],c='k')
    ax2.bar(x,5*np.ones(len(x)),width=bin_width,fc='cornflowerblue',alpha=0.3,label='5 $\sigma$')
    ax2.bar(x,-5*np.ones(len(x)),width=bin_width,fc='cornflowerblue',alpha=0.3)

    ax2.bar(x,3*np.ones(len(x)),width=bin_width,fc='yellow',alpha=0.3,label='3 $\sigma$')
    ax2.bar(x,-3*np.ones(len(x)),width=bin_width,fc='yellow',alpha=0.3)

    ax2.bar(x,1*np.ones(len(x)),width=bin_width,fc='green',alpha=0.3,label='1 $\sigma$')
    ax2.bar(x,-1*np.ones(len(x)),width=bin_width,fc='green',alpha=0.3)

    ax2.errorbar(x,(custom_h-dir_h)/dir_err,(custom_err)/dir_err,c='lightslategray',linewidth=2)

    ax2.errorbar(x,(NN_h-dir_h)/dir_err,(NN_err)/dir_err,c='r',linewidth=2.6)

    ax2.legend()
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('$ (N_{TT}-N_{direct})/\sigma_{direct} $')

    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim([max(ax2.get_ylim()[0], -20), min(ax2.get_ylim()[1], 20)])


def standard_plots(filepath, var, path2save, sample, which_leading, reg):
        
    rows = 1
    flav_list = ['b', 'c', 'l']

    fig = plt.figure(figsize=(3*6, rows*6), dpi=150)
    gs = gridspec.GridSpec(rows, 3, hspace=0.3) 
    if reg == "":
        fig_title = "Region: All"
    elif reg == "_reg1":
        fig_title = "Region: 2-3 jets"
    elif reg == "_reg2":
        fig_title = "Region: >=4 jets"
    plt.suptitle(sample + " - "+ fig_title)

    data = load_hist(filepath, var)
        
    xlabel_dict = {
        'mass': 'Mass [GeV]',
        'dR'  : 'deltaR',
        'pt' : 'pT [GeV]'
    }
        
    for flav_num, flav in enumerate(flav_list):
        plot_hist(data, flav, fig, gs[flav_num//3, flav_num%3], xlabel_dict[var])
        
    save_as = os.path.join(path2save, var + '_' + str(which_leading) +reg +'_atleast2jets.png')
    plt.savefig(save_as, bbox_inches='tight')





model = 'model101'
samples = ['sherpa221_zplusjets']  #['nominal_ttbar', 'sherpa_zplusjets']

for sample in samples:

    for which_leading in [0]:# , 1]:

        for reg in ['']:

            hist_path = os.path.join('/storage/agrp/nilotpal/truth_tagging/experiments/cxaod/boosted/', model, sample + '_hist_leading_' + str(which_leading) + reg + '.root')
            path2save = os.path.join('/storage/agrp/nilotpal/truth_tagging/experiments/cxaod/boosted/', model, sample + '_plots')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                standard_plots(hist_path, 'mass', path2save, sample, which_leading, reg)
                standard_plots(hist_path, 'dR', path2save, sample, which_leading, reg)
                standard_plots(hist_path, 'pt', path2save, sample, which_leading, reg)

