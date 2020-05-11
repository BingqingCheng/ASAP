"""
Functions for assessing the quality of the fits
"""

import numpy as np
from scipy.stats import spearmanr

class LC_SCOREBOARD():
    def __init__(self, train_sizes):
        self.scores = {size: [] for size in train_sizes}

    def add_score(self, Ntrain, score):
        self.scores[Ntrain].append(score)

    def dump_all(self):
        return self.scores

    def fetch_all(self):
        lc_results = {}
        for sc in score_func.keys():
            lc_results[sc] = self.fetch(sc)
        return lc_results

    def fetch(self, sc_name='RMSE'):
        Ntrains = []
        avg_scores = []
        avg_scores_error = []
        for Ntrain, score in self.scores.items():
            avg = 0.
            var = 0.
            for sc in score:
                avg += sc[sc_name]
                var += sc[sc_name] ** 2.
            avg /= len(score)
            var /= len(score)
            var -= avg ** 2.
            avg_scores.append(avg)
            avg_scores_error.append(np.sqrt(var))
            Ntrains.append(Ntrain)
        return np.stack((Ntrains, avg_scores, avg_scores_error), axis=-1)

def get_score(ypred, y):
    scores = {}
    for k, func in score_func.items():
        scores[k] = func(ypred, y)
    return scores

def get_r2(y_pred, y):
    weight = 1
    sample_weight = None
    numerator = (weight * (y - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (weight * (y - np.average(
        y, axis=0, weights=sample_weight)) ** 2).sum(axis=0, dtype=np.float64)
    output_scores = 1 - (numerator / denominator)
    return np.mean(output_scores)


def get_mae(ypred, y):
    return np.mean(np.abs(ypred - y))


def get_rmse(ypred, y):
    return np.sqrt(np.mean((ypred - y) ** 2))


def get_sup(ypred, y):
    return np.amax(np.abs((ypred - y)))


def get_spearman(ypred, y):
    corr, _ = spearmanr(ypred, y)
    return corr

score_func = dict(
    MAE=get_mae,
    RMSE=get_rmse,
    SUP=get_sup,
    R2=get_r2,
    CORR=get_spearman
)






