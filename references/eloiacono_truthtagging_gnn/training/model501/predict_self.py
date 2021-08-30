import numpy as np
import torch
import os
import time
from scipy.special import expit

import uproot
from helpers import utils

from tqdm import tqdm

def predict(net, dataloader, checkpoint_file, model_dir):
    
    try:
        utils.load_checkpoint(checkpoint_file, model_dir, net)
    except:
        print('checkpoint loading failed!!')
        return

    print('generating predictions...')
    predictions = []
    for x, _, _ in tqdm(dataloader):
        x = x.to(torch.device('cuda'))
        pred = net(x).cpu().data.numpy()
        predictions.extend(pred)

    return expit(predictions)



def split_pred(pred_path, filelist):

    ind_list = [0]
    for filepath in filelist:
        f = uproot.open(filepath)
        df = f['Nominal'].pandas.df(['jet_truthflav']).reset_index()
        ind_list.append(ind_list[-1] + len(df))

    pred_dir = os.path.join('/', *pred_path.split('/')[:-1])

    if type(pred_path) == np.ndarray:
        pred = pred_path
    else:
        pred = np.load(pred_path, allow_pickle=True)

    if len(pred) != ind_list[-1]:
        print('wrong combo of pred_path and filelist')
        return

    for num, filepath in enumerate(filelist):
        chunk_path = os.path.join(pred_dir, filepath.split('/')[-1][:-5] + '_prediction.npy')
        print(chunk_path)
        np.save(chunk_path, pred[ind_list[num]:ind_list[num+1]])





def npy2root(path2save, eff=None):

    if eff is None:
        eff = np.load(npy_path, allow_pickle=True)
    eff = eff.squeeze()

    num_entries = len(eff)
    chunk = 1024*1000

    with uproot.recreate(path2save) as f:
        f["Nominal_flatten"] = uproot.newtree({"efficiency": "double"})

        i = 0 # not to have an issue when num_events < chunks
        for i in range(chunk, num_entries, chunk):
            f["Nominal_flatten"].extend({"efficiency": eff[i-chunk:i]})
            
        if i != num_entries:
            f["Nominal_flatten"].extend({"efficiency": eff[i:num_entries]})


