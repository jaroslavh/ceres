import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display


def resultDF(labels, columns):
    if len(labels) != len(columns):
        raise AttributeError('Labels length ' + str(len(labels)) + ' does not match number of columns provided ' \
                             + str(len(columns)))
    test_len = len(columns[0])
    for i, column in enumerate(columns):
        if len(column) != test_len:
            raise AttributeError('Columns do not have the same length. Mismatch at index ' + str(i) \
                                 + ' with lenth column with len ' + str(len(column)))

    res = pd.DataFrame(columns=labels)
    for i, label in enumerate(labels):
        res[label] = columns[i]
    return res


def calculate_precision_recall(matrix: np.ndarray, labels, dataset):
    """Assumes matrix to be a np.ndarray of confusion matrix (regular matrix). """
    class_list = dataset.classes
    sizes = [dataset.get_class_train_size(cls) for cls in class_list]
    selected = [labels.count(cls) for cls in class_list]
    precisions = []
    recalls = []
    for row_index, row in enumerate(matrix):
        TP = matrix[row_index][row_index]
        FP = sum([i[row_index] for i in matrix]) - row[row_index]
        FN = sum(row) - row[row_index]
        precisions.append(TP / (TP + FP))
        recalls.append(TP / (TP + FN))
    return [class_list, sizes, selected, precisions, recalls]


def get_accuracy(matrix: np.ndarray):
    """Calculate overall accuracy for confusion matrix.
    
    :param matrix: Confusion matrix.
    :type matrix: np.ndarray
    :return: [description]
    :rtype: [type]
    """
    accuracies = []
    for row_index, row in enumerate(matrix):
        accuracies.append(matrix[row_index][row_index] / sum(row))
    return accuracies


def plot_confusion_matrices(matrices, titles, mat_labels, save_file=None):
    """Creates a plot of confusion matrices.
        Assumes matrices to be a list NxN np.ndarray. Titles to be a list of strings
            and header to be a list of tick_labels to be used for rows.
        Returns a figure that can then be plotted."""
    subplot_num = len(titles)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, subplot_num, figsize=(32, 12))
    ax_list = [ax1, ax2, ax3, ax4,  ax5, ax6]

    for index in range(subplot_num):
        ax = ax_list[index]

        if index >= len(matrices):
            ax.set_visible(False)
            continue

        matrix = matrices[index]
        tick_labels = mat_labels[index]
        ax.imshow(matrix, cmap='binary')

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_yticks(np.arange(len(tick_labels)))

        # ... and label them with the respective list entries
        ax.set_xticklabels(tick_labels, size=16)
        ax.set_yticklabels(tick_labels, size=16)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

        ax.set_xticks(np.arange(matrix.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(matrix.shape[0] + 1) - .5, minor=True)

        # Loop over data dimensions and create text annotations.
        for i in range(len(tick_labels)):
            for j in range(len(tick_labels)):
                if (matrix[i, j] == 0):
                    continue
                if i == j:
                    text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="g",
                                   weight="bold", fontsize=16)
                    continue
                else:
                    text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="r",
                                   weight="bold", fontsize=16)

        ax.set_title(titles[index], fontsize=24)
    if save_file != None:
        plt.savefig(save_file)
    return plt


def pandas_df_to_markdown_table(df):
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))
