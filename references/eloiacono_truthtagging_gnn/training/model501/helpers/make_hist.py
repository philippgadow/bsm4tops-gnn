import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import os



def plot_(var_list, data_path, pred, tag, entrystart, entrystop, performance_path=None, event_weight=True):

    '''
    Args:
        var_list   : list of variables to be used
        data_path  : path to the root file
        pred       : np.array() or path to np.array() 
        frac_start :
        frac_stop  :
    '''

    x_labels = {'jet_pt': 'jet pT [MeV]', 'jet_eta': 'eta', 'mJ': 'mJ [GeV]', 'MET': 'MET [GeV]'}
    x_range  = {0: {'jet_pt': [20_000, 500_000], 'jet_eta': [-2.8,2.8], 'mJ': [0, 250], 'MET': [240, 330]},
                1: {'jet_pt': [20_000, 250_000], 'jet_eta': [-2.8,2.8], 'mJ': [0, 250], 'MET': [240, 330]}}

    # make the df with all the variables
    f = uproot.open(data_path)
    if entrystop == None:
        entrystop = f['Nominal'].numentries
    entrystart, entrystop = int(entrystart), int(entrystop)

    if 'MET' not in var_list:
        df = f['Nominal'].pandas.df(var_list+['EventWeight', 'jet_truthflav', tag, 'custom_eff', 'MET'], entrystart=entrystart, entrystop=entrystop).reset_index()
    else:
        df = f['Nominal'].pandas.df(var_list+['EventWeight', 'jet_truthflav', tag, 'custom_eff'], entrystart=entrystart, entrystop=entrystop).reset_index()

    # take care of event weight
    if event_weight == False:
        df['EventWeight'] = np.ones([len(df),])

    # load the eff predictions
    if type(pred) != np.ndarray:
        pred = uproot.open(pred)['Nominal_flatten'].arrays(['efficiency'])[b'efficiency']
    df['pred_eff'] = pred

    # MET cut ('MET' is event var, so we can apply like this)
    df = df[df['MET'] > 250]

    for which_leading in [0, 1]:

        if which_leading is not None:
            df_tmp = df[df['subentry']==which_leading].copy()
        else:
            df_tmp = df.copy()

        df_tagged = df_tmp[df_tmp[tag]==1]

        for var in var_list:
            
            # initialize the figure
            fig = plt.figure(figsize=(20, 5))
            gs = gridspec.GridSpec(1, 4)
            title_flag = data_path.split('/')[-1][:-5].split('_')[0]


            if which_leading is not None:
                fig.suptitle(title_flag + ' (' + var + ') ' + 'sub-'*which_leading + 'leading')
            else:
                fig.suptitle(title_flag + ' (' + var + ')')


            data = []
            flav_dict = {5:'bottom', 4:'charm', 0:'light', 15:'tau'}
            for num, flav in enumerate([5,4,0,15]):

                df_direct  = df_tagged[df_tagged['jet_truthflav']==flav]
                df_all = df_tmp[df_tmp['jet_truthflav']==flav]

                weights_dir    = df_direct['EventWeight'].values
                weights_custom = df_all['custom_eff'].values * df_all['EventWeight'].values
                weights_NN     = df_all['pred_eff'].values * df_all['EventWeight'].values

                # create the hist and errorbars
                dir_h, dir_x = np.histogram(df_direct[var], weights=weights_dir, range=x_range[which_leading][var], bins=30)
                dir_errsq, _ = np.histogram(df_direct[var], weights=weights_dir*weights_dir, range=x_range[which_leading][var], bins=30)
                dir_err      = np.sqrt(dir_errsq)

                custom_h, _     = np.histogram(df_all[var], weights=weights_custom, range=x_range[which_leading][var], bins=30)
                custom_errsq, _ = np.histogram(df_all[var], weights=weights_custom*weights_custom, range=x_range[which_leading][var], bins=30)
                custom_err      = np.sqrt(custom_errsq)

                NN_h, _     = np.histogram(df_all[var], weights=weights_NN, range=x_range[which_leading][var], bins=30)
                NN_errsq, _ = np.histogram(df_all[var], weights=weights_NN*weights_NN, range=x_range[which_leading][var], bins=30)
                NN_err      = np.sqrt(NN_errsq)

                # plot
                gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, gs[num], hspace=0)

                x = 0.5*(dir_x[:-1] + dir_x[1:])
                bin_width = x[1]-x[0]

                ax1 = fig.add_subplot(gs_inner[0])
                ax1.set_title(flav_dict[flav])

                ax1.bar(x,dir_h,width=bin_width,fc='cornflowerblue',alpha=0.5,label='Direct Tag')
                ax1.errorbar(x, dir_h, dir_err, linestyle='none', c='k', marker='o', markersize=2, drawstyle="steps-mid")

                ax1.errorbar(x, custom_h, custom_err, c='k', label='Custom map', linewidth=2, drawstyle="steps-mid")
                ax1.errorbar(x, NN_h, NN_err, c='r', label='Neural Network', linewidth=2.2, drawstyle="steps-mid")

                ax1.legend()
                ax1.set_ylabel('Events')

                ax2 = fig.add_subplot(gs_inner[1])

                ax2.plot([x[0],x[-1]],[0,0],c='k')
                ax2.bar(x,5*np.ones(len(x)),width=bin_width,fc='cornflowerblue',alpha=0.3,label='5 $\sigma$')
                ax2.bar(x,-5*np.ones(len(x)),width=bin_width,fc='cornflowerblue',alpha=0.3)

                ax2.bar(x,3*np.ones(len(x)),width=bin_width,fc='yellow',alpha=0.3,label='3 $\sigma$')
                ax2.bar(x,-3*np.ones(len(x)),width=bin_width,fc='yellow',alpha=0.3)

                ax2.bar(x,1*np.ones(len(x)),width=bin_width,fc='green',alpha=0.3,label='1 $\sigma$')
                ax2.bar(x,-1*np.ones(len(x)),width=bin_width,fc='green',alpha=0.3)

                ax2.errorbar(x, (custom_h-dir_h)/dir_err, (custom_err)/dir_err, c='k', linewidth=2, drawstyle="steps-mid")
                ax2.errorbar(x, (NN_h-dir_h)/dir_err, (NN_err)/dir_err, c='r', linewidth=2.2, drawstyle="steps-mid")

                ax2.legend()
                ax2.set_xlabel(x_labels[var])
                ax2.set_ylabel('$ (N_{TT}-N_{direct})/\sigma_{direct} $')

                ax2.set_ylim([max(ax2.get_ylim()[0], -20), min(ax2.get_ylim()[1], 20)])


                # prepare the hists to be saved
                # dtype = object is unnecessary cause we can just save x instead of 'dir_x'
                # but that's how we have been storing till now. :(
                tmp = np.array([dir_x, dir_h, dir_err, custom_h, custom_err, NN_h, NN_err], dtype=object)
                data.append(tmp)

            plt.tight_layout(rect=[0, 0, 0.98, 0.95])

            if which_leading is not None:
                save_as = os.path.join(performance_path, var + '_' + 'sub'*which_leading + 'leading')
            else:
                save_as = os.path.join(performance_path, var)

            np.savez(save_as+'.npz', b=data[0], c=data[1], l=data[2])
            plt.savefig(save_as + '.png')
            plt.close()





def deltaphi(x):

    while len(np.where(x >= np.pi)[0]) > 0:
        x[np.where(x >= np.pi)[0]] -= 2*np.pi;
    while len(np.where(x < -np.pi)[0]) > 0: 
        x[np.where(x < -np.pi)[0]]+= 2*np.pi
    return x

def deltaR(eta,phi,eta2,phi2):
    deta = eta-eta2
    dphi = deltaphi(phi-phi2)
    return np.sqrt( deta*deta+dphi*dphi )

def plot_leading2tag(var, jet_tag, data_path, pred, entrystart=0, entrystop=None, performance_path=None, title_flag='', event_weight=False):

    '''
    Args:
        var: variable to be used
    '''

    x_labels = {'jet_pt': 'jet pT [MeV]', 'jet_eta': 'eta', 'mJ': 'mJ [GeV]', 'MET': 'MET [GeV]', 'deltaR': 'deltaR'}
    x_range  = {'jet_pt': [20_000, 500_000], 'jet_eta': [-2.8,2.8], 'mJ': [0, 250], 'MET': [240, 330], 'deltaR': [0,1.5]}

    # make the df with all the variables
    f = uproot.open(data_path)
    if entrystop == None:
        entrystop = f['Nominal'].numentries
    entrystart, entrystop = int(entrystart), int(entrystop)

    data = []

    if var == 'deltaR':
        df = f['Nominal'].pandas.df(['EventWeight', 'jet_truthflav', jet_tag, 'custom_eff', 'MET', 'jet_pt', 'jet_eta', 'jet_phi'],
                                   entrystart=entrystart,entrystop=entrystop).reset_index()
    else:
        df = f['Nominal'].pandas.df(['EventWeight', 'jet_truthflav', jet_tag, 'custom_eff', 'MET', var],
                                   entrystart=entrystart,entrystop=entrystop).reset_index()


    # take care of event weight
    if event_weight == False:
        df['EventWeight'] = np.ones([len(df),])


    # load the eff predictions
    if type(pred) != np.ndarray:
        pred = uproot.open(pred)['Nominal_flatten'].arrays(['efficiency'])[b'efficiency']
    df['pred_eff'] = pred

    # MET cut ('MET' is event var, so we can apply like this)
    df = df[df['MET'] > 250]

    if event_weight == False:
        df['weight'] = np.ones([len(df),])

    # need to have at least two jets
    df_leadingjets2 = df[df['subentry']==1]
    df_leadingjets1 = df[(df['subentry']==1).shift(periods=-1, fill_value=False)].copy()

    if var == 'deltaR':
        eta1, eta2 = df_leadingjets1['jet_eta'].values, df_leadingjets2['jet_eta'].values
        phi1, phi2 = df_leadingjets1['jet_phi'].values, df_leadingjets2['jet_phi'].values
        dR = deltaR(eta1, phi1, eta2, phi2)

        df_leadingjets1['deltaR'] = dR


    df_tagged1      = df_leadingjets1[(df_leadingjets1[jet_tag]==1).values * (df_leadingjets2[jet_tag]==1).values]
    df_tagged2      = df_leadingjets2[(df_leadingjets1[jet_tag]==1).values * (df_leadingjets2[jet_tag]==1).values]

    # initialize the figure
    fig = plt.figure(figsize=(15, 10))
    if title_flag == 'zplus':
        title = 'z+jets'
    else:
        title = title_flag
    fig.suptitle(title + ' (' + var + ')')
    gs = gridspec.GridSpec(2, 3)#, height_ratios=[3, 1, 3, 1]) 

    flav_dict = {5:'bottom', 4:'charm', 0:'light'}
    for num, (flav1, flav2) in enumerate([[5,5], [4,4], [0,0], [5,4], [5,0], [4,0]]):

        mask = (df_tagged1['jet_truthflav']==flav1).values * (df_tagged2['jet_truthflav']==flav2).values + (df_tagged1['jet_truthflav']==flav2).values * (df_tagged2['jet_truthflav']==flav1).values
        df_direct = df_tagged1[mask]

        maskNN = (df_leadingjets1['jet_truthflav']==flav1).values * (df_leadingjets2['jet_truthflav']==flav2).values + (df_leadingjets1['jet_truthflav']==flav2).values * (df_leadingjets2['jet_truthflav']==flav1).values
        df_NN = df_leadingjets1[maskNN]

        df_tmp = df_leadingjets2[maskNN]
        weights_dir = df_direct['EventWeight'].values
        weights_map = df_NN['custom_eff'].values * df_tmp['custom_eff'].values * df_NN['EventWeight'].values
        weights_NN = df_NN['pred_eff'].values * df_tmp['pred_eff'].values * df_NN['EventWeight'].values

        # create the hist and errorbars
        dir_h, dir_x = np.histogram(df_direct[var], weights=weights_dir, range=x_range[var], bins=30)
        dir_errsq, _ = np.histogram(df_direct[var], weights=weights_dir*weights_dir, range=x_range[var], bins=30)
        dir_err      = np.sqrt(dir_errsq)

        map_h, _     = np.histogram(df_NN[var], weights=weights_map, range=x_range[var], bins=30)
        map_errsq, _ = np.histogram(df_NN[var], weights=weights_map*weights_map, range=x_range[var], bins=30)
        map_err      = np.sqrt(map_errsq)

        NN_h, _     = np.histogram(df_NN[var], weights=weights_NN, range=x_range[var], bins=30)
        NN_errsq, _ = np.histogram(df_NN[var], weights=weights_NN*weights_NN, range=x_range[var], bins=30)
        NN_err      = np.sqrt(NN_errsq)


        d, r = num//3, num%3 

        # plot
        gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, gs[d,r], hspace=0)

        x = 0.5*(dir_x[:-1] + dir_x[1:])
        bin_width = x[1]-x[0]

        ax1 = fig.add_subplot(gs_inner[0])
        ax1.set_title(flav_dict[flav1] + ' ' + flav_dict[flav2])

        ax1.bar(x,dir_h,width=bin_width,fc='cornflowerblue',alpha=0.5,label='Direct Tag')
        ax1.errorbar(x,dir_h,dir_err,linestyle='none',c='k',marker='o',markersize=2, drawstyle="steps-mid")

        ax1.errorbar(x,NN_h,NN_err,c='r',label='Neural Network',linewidth=2.2, drawstyle="steps-mid")
        ax1.errorbar(x,map_h,map_err,c='gray',label='Custom Map', drawstyle="steps-mid")

        ax1.legend()
        ax1.set_ylabel('Events')

        ax2 = fig.add_subplot(gs_inner[1])

        ax2.plot([x[0],x[-1]],[0,0],c='k')
        ax2.bar(x,5*np.ones(len(x)),width=bin_width,fc='cornflowerblue',alpha=0.3,label='5 $\sigma$')
        ax2.bar(x,-5*np.ones(len(x)),width=bin_width,fc='cornflowerblue',alpha=0.3)

        ax2.bar(x,3*np.ones(len(x)),width=bin_width,fc='yellow',alpha=0.3,label='3 $\sigma$')
        ax2.bar(x,-3*np.ones(len(x)),width=bin_width,fc='yellow',alpha=0.3)

        ax2.bar(x,1*np.ones(len(x)),width=bin_width,fc='green',alpha=0.3,label='1 $\sigma$')
        ax2.bar(x,-1*np.ones(len(x)),width=bin_width,fc='green',alpha=0.3)

        ax2.errorbar(x,(NN_h-dir_h)/dir_err,(NN_err)/dir_err,c='r',linewidth=2.2, drawstyle="steps-mid")
        ax2.errorbar(x,(map_h-dir_h)/dir_err,(map_err)/dir_err,c='gray', drawstyle="steps-mid")

        ax2.legend()
        ax2.set_xlabel(x_labels[var])
        ax2.set_ylabel('$ (N_{TT}-N_{direct})/\sigma_{direct} $')

        ax2.set_ylim([max(ax2.get_ylim()[0], -20), min(ax2.get_ylim()[1], 20)])


        # prepare the hists to be saved
        tmp = np.array([dir_x, dir_h, dir_err, map_h, map_err, NN_h, NN_err], dtype=object)
        data.append(tmp)        

    plt.tight_layout(rect=[0, 0, 0.98, 0.95])

    np.savez(os.path.join(performance_path, title_flag+'_'+var+'_2leading'+'.npz'), bb=data[0], cc=data[1], ll=data[2], bc=data[3], bl=data[4], cl=data[5])

    if performance_path == None:
        return fig
    else:
        plt.savefig(os.path.join(performance_path, title_flag +'_'+ var + '_2leading' + '.png'))
        plt.close()

