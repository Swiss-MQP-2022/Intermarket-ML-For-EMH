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
    print(len(clf_list))
    print(len(datasets))
    for i in range(len(clf_list)):
        print(clf_list[i])
        print(datasets[i])
        y_score = clf_list[i].predict_proba(datasets[i].X_test)
        print(y_score)
        print(y_score[:,1])
        fpr, tpr, threshold = roc_curve(datasets[i].y_test, y_score[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append(roc_auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    #binary_y = label_binarize(data.y_test, classes=[-1, 0, 1])
    #n_classes = binary_y.shape[1]

    '''
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_y[:, i], y_score[:, i])
        fpr[i] = np.nan_to_num(fpr[i])
        tpr[i] = np.nan_to_num(tpr[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    '''
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

