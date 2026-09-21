import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def accuracy(y, pred):
    C = np.zeros((pred.max() + 1, y.max() + 1), dtype=np.int64)
    np.add.at(C, (pred, y), 1)
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(y)


def purity(y, pred):
    C = np.zeros((pred.max() + 1, y.max() + 1), dtype=np.int64)
    np.add.at(C, (pred, y), 1)
    return C.max(1).sum() / len(y)


def modularity(edges, deg, pred):
    m = len(edges)
    same = pred[edges[:, 0]] == pred[edges[:, 1]]
    dC = np.bincount(pred, weights=deg)
    return same.sum() / m - ((dC / (2 * m)) ** 2).sum()


def evaluate(ds, deg, pred):
    y = ds["y"]
    return dict(
        K=int(len(np.unique(pred))),
        NMI=normalized_mutual_info_score(y, pred),
        ARI=adjusted_rand_score(y, pred),
        ACC=accuracy(y, pred),
        purity=purity(y, pred),
        Q=modularity(ds["edges"], deg, pred),
    )
