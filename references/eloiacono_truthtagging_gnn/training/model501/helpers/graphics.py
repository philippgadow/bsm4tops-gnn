import matplotlib.pyplot as plt
import numpy as np
import json
import os
import torch


def plot_loss(path, title="Loss vs Epoch"):

    """ Returns a plot (loss vs epoch)
    Args:
        path: path to the model dir where metrics.json is
        title: optional title for the plot
    """

    filepath = os.path.join(path, 'metrics.json')

    if not os.path.exists(filepath):
        print("Metrics file not found!")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

        num_epochs = data['epochs_completed']

        train_losses = np.array(data['train_losses'])
        eval_losses  = np.array(data['validation_metric'])

        x_train = np.arange(0, len(train_losses), 1)
        x_val   = np.arange(0, len(train_losses), len(train_losses)/len(eval_losses)) + len(train_losses)/len(eval_losses)

        f, ax = plt.subplots()
        ax.plot(x_train, train_losses, color='red', label='train loss')
        ax.plot(x_val, eval_losses, color='blue', label='validation loss')

        xticks_loc = np.arange(0, len(train_losses)+1, len(train_losses)/num_epochs)
        my_xticks  = np.arange(0, num_epochs+1, 1)
        ax.set_xticks(xticks_loc)
        ax.set_xticklabels(my_xticks)

        ax.set_ylim([0, 2])

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title)
        plt.legend(loc="upper right")
        return ax



def draw_model_tensorboard(writer, net):
    
    dummy_input = torch.rand([1, 10, 4])

    writer.add_graph(net, (dummy_input))
    writer.close()
