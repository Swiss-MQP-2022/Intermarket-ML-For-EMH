from itertools import cycle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from utils import make_filename_safe

matplotlib.use('TkAgg')


def graph_roc(title, roc_list, legend_labels, plot_dir=r'./plots'):
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "blue", "red", "black", "yellow", "green"])

    for i, ((fpr, tpr, _), color) in enumerate(zip(roc_list, colors)):
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=lw,
            label=f'{legend_labels[i]} (area = {auc(fpr, tpr):0.2f})'
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC over {title}")
    plt.legend(loc="lower right")

    Path(plot_dir).mkdir(parents=True, exist_ok=True)  # create plots directory if doesn't exist

    plt.savefig(rf'{plot_dir}/roc-{make_filename_safe(title)}.png')
    plt.close()


