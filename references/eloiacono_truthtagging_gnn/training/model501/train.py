import torch
import numpy as np
import random

from helpers import utils
from tqdm import tqdm

from validation import get_validation_metric_loss


def train(param_dict):
    
    net             = param_dict['net']
    epochs          = param_dict['epochs']
    dataloader_list = param_dict['dataloader_list']
    dataloader_val  = param_dict['dataloader_val']
    optimizer       = param_dict['optimizer']
    lr              = param_dict['lr']
    loss_fun        = param_dict['loss_fun']
    model_dir       = param_dict['model_dir']
    checkpoint_file = param_dict['checkpoint_file']
    batch_multiplier= param_dict['batch_multiplier']
    log_after       = param_dict['log_after']   # log train loss after every 'log_after' batch
    flav2train      = param_dict['flav2train']
    flavNot2train   = param_dict['flavNot2train']
    scheduler       = param_dict['scheduler']

    if (flav2train is not None) and (flavNot2train is not None):
        print("Atleast one of the two prams - flav2train and flavNot2train must be None")
        return

    # load checkpoint
    if checkpoint_file is not None:
        utils.load_checkpoint(checkpoint_file, model_dir, net, optimizer)
    net.train()

    # update learning rate if needed (poor implementation, needs improvement)
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # dict where the model_state and the optim_state will be put
    state_dict = dict()

    # load the logged metrics
    metric_dict = {
        'epochs_completed' : 0,
        'lowest_metric'    : np.float('inf'),
        'train_losses'     : [],
        'validation_metric': []
    }

    utils.load_metrics(model_dir, metric_dict)
    epochs_completed = metric_dict["epochs_completed"]
    lowest_metric = metric_dict['lowest_metric']

    batch_losses = []


    # needed since we are allowing gradient accumulation
    optimizer.zero_grad()

    # main loop
    print('training...')
    for epoch in range(epochs_completed, epochs_completed+epochs):

        print('\nlearning rate: {:.1e}'.format(optimizer.param_groups[0]['lr']))
        counter = batch_multiplier
        log_counter = log_after

        net.train()

        # shuffle the dataloader list
        random.shuffle(dataloader_list)

        for dataloader_train in dataloader_list:

            for x, flav, target in tqdm(dataloader_train):

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

                batch_losses.append([loss.item()])
                loss.backward() 


                # update the counters
                counter -= 1; log_counter -= 1


                # update net
                if counter == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    counter = batch_multiplier


                # log train loss
                if log_counter == 0:
                    metric_dict['train_losses'].append(np.mean(batch_losses))
                    batch_losses = []
                    log_counter = log_after


            # run validation
            print('running validation...')
            validation_metric = get_validation_metric_loss(net, dataloader_val, loss_fun, flav2train, flavNot2train)
            metric_dict['validation_metric'].append(validation_metric)
            print('validation metric: {:.4f}'.format(validation_metric))

            if validation_metric < lowest_metric:
                print('best model till now')
                is_best = True
                lowest_metric = validation_metric
                metric_dict['lowest_metric'] = lowest_metric

            else:
                is_best = False


            # keep saving the model
            state_dict = dict()
            state_dict['model_dict'] = net.state_dict()
            state_dict['optim_dict'] = optimizer.state_dict()

            utils.save_checkpoint(state_dict, is_best=is_best, path=model_dir, epoch=None)
            utils.save_metrics(metric_dict, path=model_dir)


            # set net to train
            net.train()

        print('epochs completed: {}'.format(epoch+1))
        metric_dict['epochs_completed'] = epoch + 1
        utils.save_metrics(metric_dict, path=model_dir)

        # update learning rate
        if scheduler is not None:
            scheduler.step()

