"""
CUR matrix decomposition is a low-rank matrix decomposition algorithm that is explicitly expressed in a small number of actual columns and/or actual rows of data matrix.
"""

import numpy as np
import scipy.sparse.linalg as spalg
from scipy.linalg import svd, eigh, norm, pinv
from scipy.sparse.linalg import svds


def CUR_deterministic(X, n_col, error_estimate=True, costs=1):
    """
    Given rank k, find the best column and rows
    X = C U R

    Parameters
    ----------
    X: np.matrix
       input a covariance matrix 
    n_col: int
       number of column to keep
    error_estimate: bool, optional
       compute the remaining error of the CUR
    costs: float, optional
        a list of costs associated with each column

    Returns
    -------
    indices : np.array
        indices of columns to choose
    cur_error : np.array
        the error of the decomposition
    """

    RX = X.copy()
    rsel = np.zeros(n_col, dtype=int)
    rerror = np.zeros(n_col, dtype=float)

    for i in range(n_col):
        rval, RX = CUR_deterministic_step(RX, n_col - i, costs)
        rsel[i] = rval
        if error_estimate:
            rerror[i] = np.sum(np.abs(RX))
    return rsel, rerror


def CUR_deterministic_step(cov, k, costs=1):
    """ Apply (deterministic) CUR selection of k rows & columns of the
    given covariance matrix, including an orthogonalization step. Costs can be weighted if desired. """

    evc, evec = spalg.eigs(cov, k)
    # compute the weights and select the one with maximum weight
    weights = np.sum(np.square(evec), axis=1) / costs
    sel = np.argmax(weights)
    vsel = cov[sel] / np.linalg.norm(cov[sel])
    rcov = cov.copy()
    print("selected: ", sel)
    for i in range(len(rcov)):
        # Diagonalize the covariance matrix wrt the chosen column
        rcov[i] -= vsel * np.dot(cov[i].A1, vsel.A1)

    return sel, rcov


def cur_column_select(array, num, rank=None, deterministic=True, mode="sparse", weights=None, calc_error=False):
    """Select columns from a matrix according to statstical leverage score.

    Based on:
    [1] Mahoney MW, Drineas P. CUR matrix decompositions for improved data analysis. PNAS. 2009 Jan 20;106(3):697–702.


    Notes
    -----
    Which mode to use? If the matrix is not square or possibly not hermitian (or even real symmetric) then sparse is
    the only option. For hermitian matrices, the calculation of the eigenvectors with `eigh` may be faster than
    using the sparse svd method. Benchmark is needed for this, especially for descriptor kernels.


    Parameters
    ----------
    array : np.array
        Array to compute find columns of (M, N)
    num : int
        number of column indices to be returned

    rank : int, optional
        number of singular vectors to calculate for calculation of the statistical leverage scores
        default: minimum of (min(N, M) - 1, num/2) but at least one

    deterministic : bool, optional
        Usage of deterministic method or probabilistic.
        Deterministic (True): the top `num` columns are given
        Stochastic (False): use leverage scores as probabilities for choosing num

    mode : {"sparse", "dense", "hermitian"}
        mode of the singular vector calculation.
            - `sparse` (default), which is using sparse-SVD, which is expected to be robust and to solve the problem
            - `hermitian` offers a speedup for hermitian matrices (ie. real symmetric kernel matrices as well)
            - `dense` (not recommended) is using SVD, but calculated all the singular vectors and uses a number of them
            according to the rank parameter

    weights : np.array, optional
        Costs for taking each column. shape=(N,)
        The statistical leverage scores are scaled by this array

    calc_error : bool, optional
        calculate the error of the decomposition, default False
        There is a significant cost to the calculation of this, due to a semi-inverse operation needed.
        The error here is the norm of the matrix of difference between the original and the approximation obtained from
        the selected columns. Not necessarily meaningful in every situation.

    Returns
    -------
    indices : np.array
        indices of columns to choose
    cur_error :
    """

    if not isinstance(rank, int):
        # this is set heuristically
        rank = min(min(array.shape) - 1, max(1, int(num / 2)))

    # compute singular vectors -- the rest is not relevant from this calculation
    if mode == "sparse":
        # left_singular_vectors, _, right_singular_vectors = svds(array, k)  # for error calculation
        *_, right_singular_vectors = svds(array, rank)
    elif mode == "dense":
        *_, right_singular_vectors = svd(array)
        right_singular_vectors = right_singular_vectors[-rank:]
    elif mode == "hermitian":
        assert array.shape[0] == array.shape[1] and len(array.shape) == 2
        _, right_singular_vectors = eigh(array, eigvals=(array.shape[0] - rank, array.shape[0] - 1))
    else:
        raise ValueError(
            'mode {} is not valid for this calculation, use either of: "sparse", "dense", "hermitian"'.format(mode))

    # compute statistical leverage scores and scale by weight if given
    pi_j = np.sum(right_singular_vectors ** 2, axis=0) / rank
    if weights is not None:
        # 1D vector of length equal to the number of columns
        assert np.shape(array)[1] == np.shape(weights)[0] and len(np.shape(weights)) == 1
        pi_j *= weights
        pi_j /= np.sum(pi_j)  # normalise again

    if deterministic:
        indices = np.argsort(pi_j)[-num:]
    else:
        indices = np.random.choice(np.arange(array.shape[1]), size=num, replace=False, p=pi_j)

    if calc_error:
        # error of the representation, ||array -  C dot C+ dot array||
        c = array[:, indices]
        c_pinv = pinv(c)
        cur_error = norm(array - np.matmul(np.matmul(c, c_pinv), array))

        return indices, cur_error
    else:
        return indices
