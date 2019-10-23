"""
CUR matrix decomposition is a low-rank matrix decomposition algorithm that is explicitly expressed in a small number of actual columns and/or actual rows of data matrix.
"""

import numpy as np
import scipy.linalg as salg
import scipy.sparse.linalg as spalg


def cCURSelOrtho(cov, numSym, costs=1):
    """ Apply (deterministic) CUR selection of numSymm rows & columns of the
    given covariance matrix, including an orthogonalization step. Costs can be weighted if desired. """

    evc,evec = spalg.eigs(cov,numSym)
    weights = np.sum(np.square(evec),axis=1)/costs
    sel = np.argmax(weights)
    vsel = cov[sel].copy()
    vsel *= 1.0/np.sqrt(np.dot(vsel,vsel))
    ocov = cov.copy()
    for i in xrange(len(ocov)):
        # Diagonalize the covariance matrix wrt the chosen column
        ocov[i] -= vsel * np.dot(cov[i],vsel)

    return sel, ocov


def CURSelOrtho(cov, numSym, costs=1):

    """

    Parameters
    ----------
    cov
    numSym
    costs

    Returns
    -------

    """

    ocov = cov.copy()
    rsel = np.zeros(numSym, int)

    for i in xrange(numSym):
        rval, ocov = cCURSelOrtho(ocov, numSym-i, costs)
        rsel[i] = rval
    return rsel
