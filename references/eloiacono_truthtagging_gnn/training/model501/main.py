from model import JetEfficiencyNet
from data_loader import *
from train import train

from helpers.make_hist import *
from predict_self import *

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time
from pathlib import Path
import warnings
from helpers.graphics import plot_loss

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

sample = 'ttbar_nonallhad'
jet_tag = 'jet_tag_btag_DL1r'

model_dir = '/home/loiacoel/GNN/tt-with-gnn-boosted/'
Path(model_dir).mkdir(parents=True, exist_ok=True)

ind_start, step, ind_stop = 0, 100000, 100000


# hyperparameters
epochs = 8
lr = 3E-3
gamma = 0.1 #1/np.sqrt(30) 
milestones = [1, 5]

batch_size = 1000
effective_batch_size = 1000

net = JetEfficiencyNet(
    8, [512, 512, 512, 512, 512], [512, 256, 128, 50]
)
if torch.cuda.is_available():
    net = net.cuda()

# write about.json
net_str = '8, [512, 512, 512, 512, 512], [512, 256, 128, 50]'
utils.save_model_info(model_dir, net_str, epochs, lr, milestones, gamma, effective_batch_size)




# load train data
start_time = time.time()
print('loading dataset...')

filepath_train = '/home/loiacoel/GNN/tt-with-gnn-boosted/train.root'

entries = list(np.linspace(ind_start, ind_stop, 2))

ds_train_list = []; dataloader_list = []
ds_len_list = []

for num in range(len(entries)-1):

    entrystart, entrystop = entries[num], entries[num+1]
    ds_train = JetEventsDataset(filepath_train, jet_tag, entrystart=entrystart, entrystop= int(1E5))
    dataloader_train= DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=8, pin_memory=False)

    ds_train_list.append(ds_train)
    ds_len_list.append(len(ds_train))
    dataloader_list.append(dataloader_train)




# for validation
# ds_val = JetEventsDataset(filepath_train, jet_tag, entrystart=ind_start+step, entrystop=ind_stop+step)
ds_val = JetEventsDataset(filepath_train, jet_tag, entrystart=0, entrystop=ind_stop)
dataloader_val= DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=8, pin_memory=False)



# log info to terminal
print('dataset info -')
for num in range(len(dataloader_list)):
    print('\t# events in train dataset {}: {:.1f}M'.format(num, len(ds_train_list[num])/1E5))
print('\t# events in validation dataset: {:.1f}M'.format(len(ds_val)/1E5))

print('dataset loaded! (took {:.3f} seconds)'.format(time.time() - start_time))



# train
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

param_dict = {
    "net"             : net,
    "epochs"          : epochs,
    "dataloader_list" : dataloader_list,
    "dataloader_val"  : dataloader_val,
    "optimizer"       : optimizer,
    "lr"              : None,
    "loss_fun"        : nn.BCEWithLogitsLoss(reduction='mean'),
    "model_dir"       : model_dir,
    "checkpoint_file" : 'last_checkpoint',
    "batch_multiplier": int(effective_batch_size / batch_size),
    "log_after"       : int(effective_batch_size / batch_size),
    "flav2train"      : None,  # 5 or 4 or 0 for b,c and l respectively
    "flavNot2train"   : None,
    "scheduler"       : scheduler
}

# train
start_time = time.time()
train(param_dict)
print('train time: ', time.time() - start_time, ' seconds\n')

loss_plot = plot_loss(model_dir)
plt.savefig(os.path.join(model_dir, 'loss.png'))




#************************************************#
# predict and make 1 tag plots (on whole sample) #
#************************************************#

# entrystop = None

# # predict
# start_time = time.time()
# print('preping for prediction (on whole sample)...')

# filepath_train = '/storage/agrp/nilotpal/truth_tagging/data/cxaod_ntuples/boosted/' + sample + '_train.root'
# filepath = '/storage/agrp/nilotpal/truth_tagging/data/cxaod_ntuples/boosted/' + sample + '.root'

# print('loading data from {}'.format(filepath_train))
# ds_pred = JetEventsDataset(filepath_train, jet_tag, entrystart=0, entrystop=entrystop)
# dataloader= DataLoader(ds_pred, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=8, pin_memory=False)
# print('dataset loaded! (took {} seconds)'.format(time.time() - start_time))

# start_time = time.time()
# pred_path = os.path.join(model_dir, '_'.join(sample.split('/')) + '_prediction.root')

# print('predicting...')
# sorted_predictions = predict(net, dataloader, 'best_checkpoint', model_dir)
# print('converting to root...')
# npy2root(pred_path, sorted_predictions)
# print('conversion done! (took {:.3f} seconds)'.format(time.time() - start_time))


# # plot
# performance_path = os.path.join(model_dir, '_'.join(sample.split('/')) + '_plots')
# Path(performance_path).mkdir(parents=True, exist_ok=True)

# var_list = ['jet_pt', 'mJ']

# print('generating plots...')
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     plot_(var_list, filepath, pred_path, tag=jet_tag, entrystart=0, entrystop=entrystop, performance_path=performance_path, event_weight=True)

# var_list = ['mJ', 'deltaR']
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     for var in var_list:
#         plot_leading2tag(var, jet_tag, filepath, pred_path, entrystart=0, entrystop=entrystop, performance_path=performance_path, title_flag='zplus', event_weight=False)
# print('done!\n')
