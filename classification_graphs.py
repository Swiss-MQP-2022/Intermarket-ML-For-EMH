from itertools import cycle
from pathlib import Path

import numpy as np
import matplotlib
from sklearn.preprocessing import label_binarize

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


def graph_classification_reports(title, clf_list, legend_labels):
    lw = 2
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    for i in range(len(clf_list)):
        fpr, tpr, threshold = clf_list[i]
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append(roc_auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    plt.figure()
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "blue", "red", "black", "yellow", "green"])
    for i, color in zip(range(len(fpr_list)), colors):
        plt.plot(
            fpr_list[i],
            tpr_list[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(legend_labels[i], roc_auc_list[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(fr"ROC of Each Dataset for {title}")
    plt.legend(loc="lower right")
    plot_dir = r'./classification_plots'
    Path(plot_dir).mkdir(parents=True, exist_ok=True)  # create plots directory if doesn't exist
    plt.savefig(rf'{plot_dir}/{title}.png')
    plt.close


