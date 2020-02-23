"""
CUR matrix decomposition is a low-rank matrix decomposition algorithm that is explicitly expressed in a small number of actual columns and/or actual rows of data matrix.
"""

import numpy as np
import scipy.sparse.linalg as spalg


def CUR_deterministic(X, n_col, error_estimate=True, costs=1):
    """
    Given rank k, find the best column and rows
    X = C U R

    Parameters
    ----------
    X: input a covariance matrix 
    n_col: number of column to keep
    error_estimate: compute the remaining error of the CUR
    costs: a list of costs associated with each column

    Returns
    -------
    a list of selected columns
    """

    RX = X.copy()
    rsel = np.zeros(n_col, dtype=int)
    rerror = np.zeros(n_col, dtype=float)

    for i in range(n_col):
        rval, RX = CUR_deterministic_step(RX, n_col - i, costs)
        rsel[i] = rval
        if error_estimate: rerror[i] = np.sum(np.abs(RX))
    return rsel, rerror


def CUR_deterministic_step(cov, k, costs=1):
    """ Apply (deterministic) CUR selection of k rows & columns of the
    given covariance matrix, including an orthogonalization step. Costs can be weighted if desired. """

    evc, evec = spalg.eigs(cov, k)
    # compute the weights and select the one with maximum weight
    weights = np.sum(np.square(evec), axis=1) / costs
    sel = np.argmax(weights)
    vsel = cov[sel].copy()
    vsel *= 1.0 / np.sqrt(np.dot(vsel, vsel))
    rcov = cov.copy()
    for i in range(len(rcov)):
        # Diagonalize the covariance matrix wrt the chosen column
        rcov[i] -= vsel * np.dot(cov[i], vsel)

    return sel, rcov
