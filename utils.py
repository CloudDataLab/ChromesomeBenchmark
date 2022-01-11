import argparse
import csv
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_stratify(imgs_path):
    path_list = []
    for i, j, k in os.walk(imgs_path):
        for fileName in k:
            path_list += [os.path.join(i, fileName)]
    img_ids = []
    img_paths = []
    img_classes = []
    for p in path_list:
        img_ids.append(os.path.splitext(os.path.basename(p))[0])
        img_paths.append(p)
        img_classes.append(os.path.split(p)[0].split('/')[-1])
    col_name = {'id': img_ids, 'path': img_paths, 'class': img_classes}
    df = pd.DataFrame(col_name)
    return df


def parse_args_util():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='models name')
    args = parser.parse_args()
    return args


def load_resume_log(log_file_path):
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('train_loss', []),
        ('train_accuracy', []),
        ('val_loss', []),
        ('val_accuracy', []),
        ('is_best_epoch', [])
    ])
    with open(log_file_path, 'r') as f:
        r = csv.reader(f)
        fieldnames = next(r)
        csv_reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in csv_reader:
            for k, v in row.items():
                log[k].append(v)
    return log


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.norm = False

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", round(acc, 5))
        precision_avg = 0.0
        recall_avg = 0.0
        specificity_avg = 0.0
        F1_avg = 0.0

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 5)
            precision_avg += Precision
            Recall = round(TP / (TP + FN), 5)
            recall_avg += Recall
            Specificity = round(TN / (TN + FP), 5)
            specificity_avg += Specificity
            F1_avg += 2 * (Precision * Recall) / (Precision + Recall)
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print("the model precision is ", round(precision_avg / self.num_classes, 5))
        print("the model recall is ", round(recall_avg / self.num_classes, 5))
        print("the model specificity is ", round(specificity_avg / self.num_classes, 5))
        print("the model F1 is ", round(F1_avg / self.num_classes, 5))
        print(table)

    def plot(self):
        matrix = self.matrix
        fmt = '.0f'
        self.norm = False
        if self.norm:
            tmp = matrix.sum(axis=0)[:, np.newaxis]
            matrix = matrix / tmp
            fmt = '.4f'
        plt.figure(figsize=(18, 18))
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.xlabel('True Labels', fontsize=8)
        plt.ylabel('Predicted Labels', fontsize=8)
        plt.title('Confusion matrix', fontsize=8)
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = matrix[y, x]
                plt.text(x, y, format(info, fmt),
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black", fontsize=10)
        plt.tight_layout()
        plt.show()
