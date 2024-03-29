# 将kNN封装成类
import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k):
        """初始化类分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            'the size of X_train must be equal to the size of y_train'
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k "
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(X) for X in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert self._X_train.shape[1] == x.shape[0], \
            'the feature number of x must be equal to X_train'
        # 距离
        distance = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        # 距离排序索引
        nearest_index = np.argsort(distance)
        # 最近的k个点的类别
        topK_y = [self._y_train[i] for i in nearest_index[:self.k]]
        # 对结果进行预测
        votes = Counter(topK_y)
        predict = votes.most_common(1)[0][0]
        return predict

    def __repr__(self):
        return "KNN(k=%d)" % self.k
