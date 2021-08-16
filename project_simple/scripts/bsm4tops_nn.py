import numpy as np
import pandas as pd

from argparse import ArgumentParser
from os import makedirs
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn, optim
from utils import getDataFrame, cleanDataFrame, augmentDataFrame
from utils import visualizeDataFrame, visualizeLossAcc, visualizeROC


def getArgumentParser():
    parser = ArgumentParser()
    parser.add_argument('inputFile')
    parser.add_argument('--dnn', action='store_true', help='Run fully connected deep neural network and evaluate its performance.')
    parser.add_argument('--plot', action='store_true', help='Create overview plots.')
    return parser


def get_device():
    """Determine supported device: use GPU if avaliable, otherwise use CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device


def runDNNClassifier(X_train, X_test, y_train, y_test):
    """Run simple deep-neural-network-based classification.
    
    Args:
        X_train (pandas dataframe): training data, contains features for training
        X_test (pandas dataframe): test data, contains features for evaluating performance
        y_train (pandas series): training targets, contains targets for training
        y_test (pandas series): test targets, contains true targets to evaluate performance of prediction
    """

    device = get_device()

    # access input data
    X_train = X_train.astype(np.float32).values
    y_train = y_train.astype(np.float32)

    # preprocessing
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    # convert data into pytorch dataset
    from torch.utils.data import Dataset, DataLoader
    class dataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.length = self.x.shape[0]
     
        def __getitem__(self,idx):
            return self.x[idx],self.y[idx]
        def __len__(self):
            return self.length

    # define training dataset and training dataloader
    trainingset = dataset(X_train, y_train)
    trainingloader = DataLoader(trainingset, batch_size=512, shuffle=True)

    # define the network
    from torch import nn
@@ -137,33 +137,77 @@ def plotEventVariables(tops_p4):
    """

    # define histograms
    h_mass_resonance = hist.Hist.new.Reg(30, 1000, 3200).Int64()
    h_mass_spectator = hist.Hist.new.Reg(30, 1000, 3200).Int64()
    h_mass_resonance = hist.Hist.new.Reg(30, 0, 3200).Int64()
    h_mass_spectator = hist.Hist.new.Reg(30, 0, 3200).Int64()
    h_dRtt_resonance = hist.Hist.new.Reg(30, 0, 6).Int64()
    h_dRtt_spectator = hist.Hist.new.Reg(30, 0, 6).Int64()
    h_dist_resonance = hist.Hist.new.Reg(30, 0, 6).Int64()

    from torch.nn import functional as F

    class DNN(nn.Module):
        """Deep neural network (very basic) implemented in pytorch.
        Network architecture:
        - input features: 4 (pt, eta, phi, m)
        - first layer: 3 nodes
        - second layer: 5 nodes
        - output layer: 1 node (with probability for being a resonance top quark)
        """
        def __init__(self, input_shape):
            super(DNN,self).__init__()
            layer_one_neurons = 32
            layer_two_neurons = 32
            self.fc1 = nn.Linear(input_shape, layer_one_neurons)
            self.fc2 = nn.Linear(layer_one_neurons,layer_two_neurons)
            self.fc3 = nn.Linear(layer_two_neurons, 1)

        def forward(self,x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    # hyper parameters (used in training)
    learning_rate = 0.01
    
    epochs = 300

    # define model
    model = DNN(input_shape=X_train.shape[1])
    # define optimizer and loss function used in training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    # forward loop
    losses = []
    accur = []
    for i in tqdm(range(epochs)):
        for j, (x_training, y_training) in enumerate(trainingloader):
            # calculate output
            output = model(x_training)
         
            # calculate loss
            loss = loss_function(output, y_training.reshape(-1, 1))
         
            # calculate accuracy
            predicted = model(torch.tensor(X_train, dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().numpy().round() == y_train).mean()
            # perform backpropagation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # collect loss + accuracy, give update to output
        losses.append(float(loss))
        accur.append(acc)
        if i%10 == 0:
            print(f"epoch {i}\tloss : {loss}\t accuracy : {acc}")


    # visualise loss and accuracy during training
    visualizeLossAcc(losses, 'Loss', join('plots', 'dnn_loss_train.png'))
    visualizeLossAcc(accur, 'Accuracy', join('plots', 'dnn_accur_train.png'))

    # prediction using test data
    X_test = X_test.astype(np.float32).values
    X_test = sc.fit_transform(X_test)
    X_test = torch.tensor(X_test)
    preds = model(X_test).detach().numpy()

    # plot ROC curve
    makedirs('plots', exist_ok=True)
    visualizeROC(preds, y_test, join('plots', 'dnn_ROC_test.png'))


def main():
    # get command line arguments
    args = getArgumentParser().parse_args()

    # parse input file into a pandas dataframe
    df = getDataFrame(args.inputFile)
    df = cleanDataFrame(df)

    # augment dataframe with additional variables
    df = augmentDataFrame(df)

    # shuffle data frame
    df = shuffle(df)

    # visualise content of dataframe
    if args.plot:
        visualizeDataFrame(df, 'plots')

    # split dataset into a part containing the features used for prediction: X
    # and a part containing the targets of the prediction (the true values): y
    target_names = ['resonance']
    y = df.pop('resonance').values
    feature_names = ['Particle.PT', 'Particle.Eta', 'Particle.Phi', 'Particle.M']
    X = df

    # split in training and testing part
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # run the DNN classifier
    if args.dnn:
        runDNNClassifier(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
