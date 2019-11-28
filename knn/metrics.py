import numpy as np


def accuracy_score(y_truth, y_predict):
    """计算y_truth和y_predict之间的准确率"""
    assert y_truth.shape[0] == y_predict.shape[0], \
        'the size of y_true must be equal to the size of y_predict'
    return sum(y_truth == y_predict) / len(y_truth)
