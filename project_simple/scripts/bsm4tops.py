import numpy as np
import pandas as pd

from argparse import ArgumentParser
from os import makedirs
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils import getDataFrame, cleanDataFrame, augmentDataFrame
from utils import visualizeDataFrame, visualizeBDTScore, visualizeROC


def getArgumentParser():
    parser = ArgumentParser()
    parser.add_argument('inputFile')
    parser.add_argument('--simple', action='store_true', help='Run simple classifier and evaluate its performance.')
    parser.add_argument('--knn', action='store_true', help='Run k-nearest-neighbours classifier and evaluate its performance.')
    parser.add_argument('--bdt', action='store_true', help='Run BDT classifier and evaluate its performance.')
    parser.add_argument('--skip_plotting', action='store_true', help='Do not create overview plots (save some time when running).')
    return parser


def runSimpleClassifier(X_train, X_test, y_train, y_test):
    """Run simple rule-based classification.
    Instead of using machine-learning, we use human-learning and come up 

    
    Args:
        X_train (pandas dataframe): training data, contains features for training
        X_test (pandas dataframe): test data, contains features for evaluating performance
        y_train (pandas series): training targets, contains targets for training
        y_test (pandas series): test targets, contains true targets to evaluate performance of prediction
    """

    # we don't train with the simple classifier:
    # https://www.youtube.com/watch?v=_MqfxuAgME4

    # based of our knowledge on the PT distribution we 
    # just define a PT cut and directly go to the testing data for evaluation

    # TODO: the PT cut is VERY BAD. Try to do better by looking at the pt distributions of top quarks
    #       from the resonance and top quarks from associated production and come up with a better
    #       value

    pt_cut = 100.
    preds = (X_test['Particle.PT'] > pt_cut).astype(int).to_numpy()

    # TODO: check if adding a second requirement, e.g. on eta could improve the selection
    #       hint: you can use logical operations to chain cuts
    #             (a['some_column']==some_number) & (a['some_other_column']==some_other_number)


    # TODO: perhaps you can come up with yet another observable?
    #       a candidate might be the distance of a quark to another quark
    #       this variable needs to be added to the dataframe.
    #       have a look at scripts/utils/io -> augmentDataFrame()
    #       and https://stackoverflow.com/questions/26886653/pandas-create-new-column-based-on-values-from-other-columns-apply-a-function-o


    # plot ROC curve
    makedirs('plots', exist_ok=True)
    visualizeROC(preds, y_test, join('plots', 'simple_ROC_test.png'))


def runNearestNeighbours(X_train, X_test, y_train, y_test):
    """Run nearest neighbours algorithm for classification.
    More details about the algorithm:
    https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification
    
    Args:
        X_train (pandas dataframe): training data, contains features for training
        X_test (pandas dataframe): test data, contains features for evaluating performance
        y_train (pandas series): training targets, contains targets for training
        y_test (pandas series): test targets, contains true targets to evaluate performance of prediction
    """
    from sklearn.neighbors import KNeighborsClassifier


    # set up classifier
    knn = KNeighborsClassifier()

    # train classifier
    knn.fit(X_train, y_train)

    # plot ROC curve
    probs = knn.predict_proba(X_test)
    preds = probs[:,1]
    makedirs('plots', exist_ok=True)
    visualizeROC(preds, y_test, join('plots', 'knn_ROC_test.png'))


def runBDT(X_train, X_test, y_train, y_test):
    """Run Boosted Decision Tree algorithm for classification.
    More details about the algorithm:
    https://scikit-learn.org/stable/modules/ensemble.html#adaboost
    
    Args:
        X_train (pandas dataframe): training data, contains features for training
        X_test (pandas dataframe): test data, contains features for evaluating performance
        y_train (pandas series): training targets, contains targets for training
        y_test (pandas series): test targets, contains true targets to evaluate performance of prediction
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    # set up classifier
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             algorithm="SAMME",
                             n_estimators=200)
    # train classifier
    bdt.fit(X_train, y_train)

    # plot BDT score for training dataset
    twoclass_output_train = bdt.decision_function(X_train)
    class_names = ['spectator', 'resonance']
    plot_colors = ['blue', 'orange']
    plot_range = (twoclass_output_train.min(), twoclass_output_train.max())
    makedirs('plots', exist_ok=True)
    visualizeBDTScore(twoclass_output_train, y_train, class_names, plot_range, plot_colors, join('plots', 'bdt_score_train.png'))

    # plot BDT score for test dataset
    twoclass_output_test = bdt.decision_function(X_test)
    plot_range = (twoclass_output_test.min(), twoclass_output_test.max())
    visualizeBDTScore(twoclass_output_test, y_test, class_names, plot_range, plot_colors, join('plots', 'bdt_score_test.png'))

    # plot ROC curve
    probs = bdt.predict_proba(X_test)
    preds = probs[:,1]
    visualizeROC(preds, y_test, join('plots', 'bdt_ROC_test.png'))


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
    if not args.skip_plotting:
        visualizeDataFrame(df, 'plots')

    # split dataset into a part containing the features used for prediction: X
    # and a part containing the targets of the prediction (the true values): y
    target_names = ['resonance']
    y = df.pop('resonance').values
    feature_names = ['Particle.PT', 'Particle.Eta', 'Particle.Phi', 'Particle.M']
    X = df

    # split in training and testing part
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # run the simple classifier
    if args.simple:
        runSimpleClassifier(X_train, X_test, y_train, y_test)

    # run nearest neighbours classifier
    if args.knn:
        runNearestNeighbours(X_train, X_test, y_train, y_test)

    # run the BDT classifier
    if args.bdt:
        runBDT(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
