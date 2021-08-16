import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from os import makedirs
from os.path import join


# Plotting helper functions


def visualizeDataFrame(df, outputdir):
    # create output directory
    makedirs(outputdir, exist_ok=True)

    # make overview plot
    sns_plot = sns.pairplot(df, hue='resonance', height=2.5)
    plt.savefig(join(outputdir, 'overview.png'))



def visualizeBDTScore(twoclass_output, y, class_names, plot_range, plot_colors, outputname):
    fig, ax = plt.subplots()
    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(twoclass_output[y == i],
                 bins=10,
                 range=plot_range,
                 facecolor=c,
                 label='Class %s' % n,
                 alpha=.5,
                 edgecolor='k')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.2))
    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score')
    plt.title('Decision Scores')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    fig.savefig(outputname)


def visualizeLossAcc(values, title, outputname):
    fig, ax = plt.subplots()
    plt.plot(values)
    plt.title(f'{title} vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{title}')
    plt.tight_layout()
    fig.savefig(outputname)


def visualizeROC(preds, y, outputname):
    import sklearn.metrics as metrics
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    fig, ax = plt.subplots()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    fig.savefig(outputname)
