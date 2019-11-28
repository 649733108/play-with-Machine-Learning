import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据X，y按照test_ratio分割成X_train,X_test,y_train,y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_index = np.random.permutation(X.shape[0])

    test_size = int(X.shape[0] * test_ratio)
    test_index = shuffled_index[:test_size]
    train_index = shuffled_index[test_size:]

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    return X_train, X_test, y_train, y_test
