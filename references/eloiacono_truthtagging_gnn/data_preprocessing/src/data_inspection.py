import uproot
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os
import matplotlib
matplotlib.rcParams.update({'font.size': 12})


def flav_composition(filepath, title, save_as):
    
    f = uproot.open(filepath)
    df = f['Nominal'].pandas.df(['jet_truthflav', 'jet_tag_btag_DL1r']).reset_index()
        
    total_jets  = int(1E&)  # len(df)

    total_bjets = (df['jet_truthflav'] == 5).sum()
    total_cjets = (df['jet_truthflav'] == 4).sum()
    total_ljets = (df['jet_truthflav'] == 0).sum()
    total_oths  = total_jets - (total_bjets + total_cjets + total_ljets)

    b_tagged_bjets = ((df['jet_truthflav'] == 5) & (df['jet_tag_btag_DL1r'] == 1)).sum()
    b_tagged_cjets = ((df['jet_truthflav'] == 4) & (df['jet_tag_btag_DL1r'] == 1)).sum()
    b_tagged_ljets = ((df['jet_truthflav'] == 0) & (df['jet_tag_btag_DL1r'] == 1)).sum()
    
    fig = plt.figure(figsize=(7, 5), dpi=150)
    fig.suptitle(title, y=1.2)
    gs = gridspec.GridSpec(1, 1, hspace=1)
    
    # first pie
    labels = 'tagged b jets', 'untagged b jets', 'tagged c jets', 'untagged c jets', 'tagged light jets', 'untagged light jets', 'others'
    sizes = [b_tagged_bjets, total_bjets-b_tagged_bjets, b_tagged_cjets, total_cjets-b_tagged_cjets, b_tagged_ljets, total_ljets-b_tagged_ljets, total_oths]
    colors = ['cyan', 'darkturquoise', 'lightgreen', 'yellowgreen', 'mediumpurple', 'slateblue', 'peru']
    explode = (0.05, 0, 0.05, 0, 0.05, 0, 0)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('btag (DL1r, 70%)', y=1.15)

    patches, _, _ = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode,
            shadow=True, pctdistance=0.7, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    labels = ['{:,}'.format(x) for x in sizes]
    legend = ax1.legend(patches, labels, loc='upper right', bbox_to_anchor=(1.05, 1.2),
           fontsize=12)

    renderer = fig.canvas.get_renderer()
    shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
    for t in legend.get_texts():
        t.set_ha('right') # ha is alias for horizontalalignment
        t.set_position((shift,0))
    
    plt.savefig(save_as, bbox_inches='tight')





def var_distribution(filepath, title, save_as):

    f = uproot.open(filepath)

    jetvar   = ['jet_pt', 'jet_eta', 'jet_phi'] #, 'H_m1', 'H_pT1', 'H_eta1', 'H_phi1']
    eventvar = ['mJ', 'MET']


    count_jetvar, count_eventvar, count_var = len(jetvar), len(eventvar), len(jetvar) + len(eventvar)

    xlabel = {
        'jet_pt': 'jet pT [MeV]', 'jet_eta': 'jet eta', 'jet_phi': 'jet_phi',
      #  'H_m1': 'H_mass [MeV]', 'H_pT1': 'H_pT [GeV]', 'H_eta1': 'H_eta', 'H_phi1': 'H_phi',
        'pdg_parton': 'parton id', 'mJ': 'mJ [GeV]', 'MET': 'MET [GeV]'
    }

    log = {
        'jet_pt': True, 'jet_eta': False, 'jet_phi': False,
        'H_m1': True, 'H_pT1': True, 'H_eta1': False, 'H_phi1': False,
        'pdg_parton': True, 'mJ': True, 'MET': True
    }

    flav_dict = {5:'bottom', 4:'charm', 0:'light', 15:'tau'}


    df_jetvar = f['Nominal'].pandas.df(jetvar+['jet_truthflav']).reset_index()
    df_eventvar = f['Nominal'].pandas.df(eventvar).reset_index()

    fig = plt.figure(figsize=(4*9, count_var*6), dpi=150)
    fig.suptitle(title, y=0.9)
    gs = gridspec.GridSpec(count_var, 1, hspace=1)

    gs_num = 0
    for var in jetvar:

        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 4, gs[gs_num], wspace=0.5)
        gs_num += 1

        # flav loop
        for num_flav, flav in enumerate([5, 4, 0, 15]):
            ax = fig.add_subplot(gs_inner[num_flav])
            df_jetvar[df_jetvar['jet_truthflav']==flav][var].plot.hist(ax=ax, bins=30, logy=log[var])

            ax.set_title('{} ({})'.format(var, flav_dict[flav]))
            ax.set_xlabel(xlabel[var])
            if log[var] == True:
                ax.set_ylabel('log(#)')
            else:
                ax.set_ylabel('#')

            



    for var in eventvar:

        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 4, gs[gs_num], wspace=0.5)
        gs_num += 1

        ax = fig.add_subplot(gs_inner[0])

        df_eventvar[var].plot.hist(ax=ax, bins=30, logy=log[var])

        ax.set_title(var)
        ax.set_xlabel(xlabel[var])
        if log[var] == True:
            ax.set_ylabel('log(#)')
        else:
            ax.set_ylabel('#')


    plt.savefig(save_as, bbox_inches='tight')







# sample   = 'sherpa221/zplusjets'
# filepath = '/storage/agrp/nilotpal/truth_tagging/data/cxaod_ntuples/boosted/' + sample + '.root'

# title    = 'Z+jets (flavor composition)'
# save_as  = '/srv01/agrp/nilotpal/projects/truth_tagging/cxaod_ntuples/boosted/data_preprocessing/_ntuples_info/zplusjets_flav_comp.png'
# flav_composition(filepath, title, save_as)

# title    = 'Z+jets (variable distribution)'
# save_as  = '/srv01/agrp/nilotpal/projects/truth_tagging/cxaod_ntuples/boosted/data_preprocessing/_ntuples_info/zplusjets_var_dist.png'
# var_distribution(filepath, title, save_as)




sample   = 'ttbar_nonallhad'
filepath = '/eos/user/e/eloiacon/SWAN_projects/GNN/ATLAS_GNN/latest_samples/ttbar_nonallhad.root'

title    = 'ttbar (flavor composition)'
save_as  = '/eos/user/e/eloiacon/SWAN_projects/GNN/ATLAS_GNN/latest_samples/ttbar_flav_comp.png'
flav_composition(filepath, title, save_as)

title    = 'ttbar (variable distribution)'
save_as  = '/eos/user/e/eloiacon/SWAN_projects/GNN/ATLAS_GNN/latest_samples/ttbar_var_dist.png'
var_distribution(filepath, title, save_as)

