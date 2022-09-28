from itertools import cycle

import numpy as np
import matplotlib
from sklearn.preprocessing import label_binarize

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


def graph_classification_reports(estimator_name, clf_list, datasets):
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

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "blue", "red", "black", "yellow", "green"])
    for i, color in zip(range(len(fpr_list)), colors):
        plt.plot(
            fpr_list[i],
            tpr_list[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(datasets[i].name, roc_auc_list[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(fr"ROC of Each Dataset for {estimator_name}")
    plt.legend(loc="lower right")
    plt.show()

