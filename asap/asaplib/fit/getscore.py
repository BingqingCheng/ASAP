import numpy as np
from scipy.stats import spearmanr


def get_r2(y_pred, y_true):
    weight = 1
    sample_weight = None
    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0, dtype=np.float64)
    output_scores = 1 - (numerator / denominator)
    return np.mean(output_scores)


def get_mae(ypred, y):
    return np.mean(np.abs(ypred-y))


def get_rmse(ypred, y):
    return np.sqrt(np.mean((ypred-y)**2))


def get_sup(ypred, y):
    return np.amax(np.abs((ypred-y)))


def get_spearman(ypred, y):
    corr,_ = spearmanr(ypred,y)
    return corr

score_func = dict(
    MAE=get_mae,
    RMSE=get_rmse,
    SUP=get_sup,
    R2=get_r2,
    CORR=get_spearman
)


def get_score(ypred,y):
    scores = {}
    for k,func in score_func.items():
        scores[k] = func(ypred,y)
    return scores
