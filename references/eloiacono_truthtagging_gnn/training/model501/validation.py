import numpy as np
import torch
import uproot
from scipy.special import expit
import warnings
from tqdm import tqdm


def get_validation_metric_loss(net, dataloader, loss_fun, flav2train, flavNot2train):

    net.eval()

    losses = []
    for x, flav, target in tqdm(dataloader):

        # use GPU if available
        if torch.cuda.is_available():
            x, flav, target = x.to(torch.device('cuda')), flav.cuda(), target.cuda()

        output = net(x)

        if flav2train is not None:
            mask = torch.where(flav.view(-1)==flav2train)
            loss = loss_fun(output[mask], target[mask])
        elif flavNot2train is not None:
            mask = torch.where(flav.view(-1)!=flavNot2train)
            loss = loss_fun(output[mask], target[mask])
        else:
            loss = loss_fun(output, target)

        losses.append([loss.item()])

    return np.mean(losses)






def get_validation_metric_old(net, dataloader, df_validation):

    net.eval()
    predictions = []
    for x, _, _ in dataloader:
        x = x.to(torch.device('cuda'))
        pred = net(x).cpu().data.numpy()
        predictions.extend(pred)

    predictions = expit(predictions).squeeze()


    var_list = ['jet_pt', 'jet_eta']
    x_range  = {'jet_pt': [20_000, 250_000], 'jet_eta': [-2.8,2.8]}


    # finalise the df to work with
    df = df_validation.copy()
    df['pred_eff'] = predictions


    metric = 0
    for which_leading in [0, 1, 2, 3, 4, 5]:

        if which_leading is not None:
            df_tmp = df[df['subentry']==which_leading].copy()
        else:
            df_tmp = df.copy()

        df_tagged = df_tmp[df_tmp["jet_tag"]==1]

        for var in var_list:

            data = []
            flav_dict = {5:'bottom', 4:'charm', 0:'light', 15:'tau'}
            for num, flav in enumerate([5,4,0]):

                df_direct  = df_tagged[df_tagged['jet_truthflav']==flav]
                df_all = df_tmp[df_tmp['jet_truthflav']==flav]

                weights_dir    = df_direct['EventWeight'].values
                weights_NN     = df_all['pred_eff'].values * df_all['EventWeight'].values

                # create the hist and errorbars
                dir_h, dir_x = np.histogram(df_direct[var], weights=weights_dir, range=x_range[var], bins=30)
                dir_errsq, _ = np.histogram(df_direct[var], weights=weights_dir*weights_dir, range=x_range[var], bins=30)
                dir_err      = np.sqrt(dir_errsq)

                NN_h, _     = np.histogram(df_all[var], weights=weights_NN, range=x_range[var], bins=30)
                NN_errsq, _ = np.histogram(df_all[var], weights=weights_NN*weights_NN, range=x_range[var], bins=30)
                NN_err      = np.sqrt(NN_errsq)


                # find chisq and ignore the invalid value
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chisq_NN  = ((NN_h - dir_h)**2 / dir_h)
                chisq_NN = chisq_NN[(~np.isnan(chisq_NN)) & (chisq_NN!=np.float('+inf')) & (chisq_NN!=np.float('-inf'))]

                metric += chisq_NN.sum()

    return metric



