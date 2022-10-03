from itertools import cycle
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm

from utils import make_filename_safe, DataSplit
from constants import METRICS

matplotlib.use('TkAgg')


def graph_roc(title, roc_list, legend_labels, plot_dir):
    """
    Generate an ROC plot
    :param title: Title of the plot
    :param roc_list: roc data generated by roc_curve
    :param legend_labels: labels for the legend. Correspond to each model/dataset in plot
    :param plot_dir: directory to save plot to
    """
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 'red', 'black', 'yellow', 'green'])

    for i, ((fpr, tpr, _), color) in enumerate(zip(roc_list, colors)):
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=lw,
            label=f'{legend_labels[i]} (area = {auc(fpr, tpr):0.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC over {title}')
    plt.legend(loc='lower right')

    plt.savefig(rf'{plot_dir}/roc-{make_filename_safe(title)}.png')
    plt.close()


def graph_all_roc(data, plot_dir=r'./out/plots'):
    """
    Generate all ROC graphs
    :param data: dataframe containing metric data
    :param plot_dir: directory to save plot to
    """
    print('Generating ROC graphs...')

    Path(plot_dir).mkdir(parents=True, exist_ok=True)  # create plots directory if doesn't exist

    # Generate ROC w.r.t. model plots
    for model_name, model in tqdm(data['roc', DataSplit.TRAIN].groupby(level=0)):
        graph_roc(f'model: {model_name}', model.to_numpy(), model.index.get_level_values(1).tolist(), plot_dir)
    # Generate ROC w.r.t. dataset plots
    for data_name, dataset in tqdm(data['roc', DataSplit.TRAIN].groupby(level=1)):
        graph_roc(f'dataset: {data_name}', dataset.to_numpy(), dataset.index.get_level_values(0).tolist(), plot_dir)


def save_metrics(data: pd.DataFrame, out_dir=r'./out'):
    """
    Generate CSVs of desired metrics
    :param data: dataframe containing metric data
    :param out_dir: directory to save CSVs
    """
    print('Saving metrics...')

    Path(out_dir).mkdir(parents=True, exist_ok=True)  # create plots directory if doesn't exist

    # Save metrics to csv
    metric_reports = data['classification report'].unstack(level=0).swaplevel(0, 1, axis=1)
    for metric_name, metric in METRICS.items():
        metric_data = metric_reports.applymap(
            lambda x: x[metric[0]][metric[1]] if isinstance(metric, tuple) else x[metric])
        metric_data.to_csv(rf'./out/{make_filename_safe(metric_name)}.csv')
