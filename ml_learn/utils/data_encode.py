import numpy as np

def one_hot(x, C):
    N = len(x) if isinstance(x, list) else x.shape[0]
    X = np.zeros((N, C))
    X[range(N), x] = 1
    return X
