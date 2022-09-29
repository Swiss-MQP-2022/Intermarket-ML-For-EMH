from itertools import cycle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import auc

matplotlib.use('TkAgg')


def graph_classification_reports(title, clf_list, legend_labels, plot_dir=r'./plots'):
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
            label=f'{legend_labels[i]} (area = {roc_auc_list[i]:0.2f})'
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC over {title}")
    plt.legend(loc="lower right")

    Path(plot_dir).mkdir(parents=True, exist_ok=True)  # create plots directory if doesn't exist

    plt.savefig(rf'{plot_dir}/roc-{title.replace(" ", "_").replace(":", "")}.png')
    plt.close()


