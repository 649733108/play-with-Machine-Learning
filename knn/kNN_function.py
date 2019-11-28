import numpy as np
from math import sqrt
from collections import Counter


def kNN_classifier(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], 'k must be valid'
    assert X_train.shape[0] == y_train.shape[0], \
        'the size of X_train must be equal to the size of y_train'
    assert X_train.shape[1] == x.shape[0], \
        'the feature number of x must be equal to X_train'

    # 距离
    distance = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
    # 距离排序索引
    nearest_index = np.argsort(distance)
    # 最近的k个点的类别
    topK_y = [y_train[i] for i in nearest_index[:k]]
    # 对结果进行预测
    votes = Counter(topK_y)
    predict = votes.most_common(1)[0][0]
    return predict
