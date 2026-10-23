"""Original large-null numerical kernels. Private activation inputs are unavailable."""
import numpy as np
from sklearn.linear_model import LogisticRegression
SEED = 20260803
LOGISTIC_C = 0.1
MAX_ITER = 2000

def dim_direction(X, y):
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    direction = mu1 - mu0
    direction = direction / np.linalg.norm(direction)
    return direction, float(0.5 * (mu1 + mu0) @ direction)


def i_b(scores_eval, scores_select, g):
    z = (scores_eval - float(scores_select.mean())) / float(scores_select.std(ddof=1))
    yev = g["yev"]
    return float(np.mean([z[m & (yev == 1)].sum() - z[m & (yev == 0)].sum()
                          for m in g["board_masks"]]))


def evaluate(ytr, g, seed):
    d, mid = dim_direction(g["Xtr"], ytr)
    dim = i_b(g["Xev"] @ d - mid, g["Xse"] @ d - mid, g)
    clf = LogisticRegression(C=LOGISTIC_C, solver="liblinear", dual=True,
                             max_iter=MAX_ITER, random_state=seed).fit(g["Xtr"], ytr)
    return dim, i_b(clf.decision_function(g["Xev"]), clf.decision_function(g["Xse"]), g)


def flip(y, groups, rng):
    out = y.astype(np.int8).copy()
    for grp in np.unique(groups):
        if bool(rng.integers(0, 2)):
            mask = groups == grp
            out[mask] = 1 - out[mask]
    return out

