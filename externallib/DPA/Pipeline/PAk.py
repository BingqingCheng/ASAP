# The pointwise-adaptive k-NN density estimator.
#
# Author: Maria d'Errico <mariaderr@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, DensityMixin, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from math import log, sqrt, exp, lgamma, pi, pow
from Pipeline.twoNN import twoNearestNeighbors
from Pipeline import _PAk

VALID_METRIC = ['precomputed', 'euclidean']
VALID_DIM = ['auto', 'twoNN']

def _PointAdaptive_kNN(distances, indices, k_max=1000, D_thr=23.92812698, dim=None):
    """Main function implementing the Pointwise Adaptive k-NN density estimator.

    Parameters
    ----------
    distances: array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points. The point is included. 
  
    indices : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points. The point is included.

    k_max : int, default=1000
        The maximum number of nearest-neighbors considered by the procedure that returns the
        largest number of neighbors ``\hat{k}`` for which the condition of constant density 
        holds, within a given level of confidence. If the number of points in the sample N is 
        less than the default value, k_max will be set automatically to the value ``N/2``.
    
    D_thr : float, default=23.92812698
        Set the level of confidence. The default value corresponds to a p-value of 
        ``10**{-6}`` for a ``\chiˆ2`` distribution with one degree of freedom.

    dim : int
        Intrinsic dimensionality of the sample.

    Results
    -------
    densities : array [n_samples]
        The logarithm of the density at each point.
    
    err_densities : array [n_samples]
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat : array [n_samples]
        The optimal number of neighbors for which the condition of constant density holds.

    dc : array [n_sample]
        The radius of the optimal neighborhood for each point.
    
    References
    ----------
    # TODO 

    """
    # The adaptive k-Nearest Neighbor density estimator
    k_hat, dc, densities, err_densities = _PAk.get_densities(dim, distances, k_max, D_thr, indices)

    return densities, err_densities, k_hat, dc


class PointAdaptive_kNN(BaseEstimator, DensityMixin):
    """Class definition for the Pointwise Adaptive k-NN density estimator.

    Parameters
    ----------
    k_max : int, default=1000
        The maximum number of nearest-neighbors considered by the procedure that returns the
        largest number of neighbors ``\hat{k}`` for which the condition of constant density 
        holds, within a given level of confidence. If the number of points in the sample N is 
        less than the default value, k_max will be set automatically to the value ``N/2``.
    
    D_thr : float, default=23.92812698
        Set the level of confidence. The default value corresponds to a p-value of 
        ``10**{-6}`` for a ``\chiˆ2`` distribution with one degree of freedom.

    metric : string, or callable
        The distance metric to use. 
        If metric is a string, it must be one of the options allowed by 
        scipy.spatial.distance.pdist for its metric parameter, or a metric listed in 
        pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is "precomputed", X is assumed to
        be a distance matrix. Alternatively, if metric is a callable function, it is 
        called on each pair of instances (rows) and the resulting value recorded. The 
        callable should take two arrays from X as input and return a value indicating 
        the distance between them. Default is 'euclidean'.

    dim_algo : string, or callable
        Method for intrinsic dimensionality calculation. If dim_algo is "auto", dim is assumed to be
        equal to n_samples. If dim_algo is a string, it must be one of the options allowed by VALID_DIM. 

    blockAn : bool, default=True
        This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
        If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant dimensions 
        as a function of the block size. This allows to study the stability of the estimation with respect to
        changes in the neighborhood size, which is crucial for ID estimations when the data lie on a manifold perturbed 
        by a high-dimensional noise. 

    block_ratio : int, default=20
        This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
        Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False, block_ratio is ignored.

    frac : float, default=1
        This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
        Define the fraction of points in the data set used for ID calculation. By default the full data set is used.   

    dim : int
        Intrinsic dimensionality of the sample. If dim is provided, dim_algo is ignored.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    distances_ : array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points.

    indices_ : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points.
 
    densities_ : array [n_samples]
        The logarithm of the density at each point.
    
    err_densities_ : array [n_samples]
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat_ : array [n_samples]
        The optimal number of neighbors for which the condition of constant density holds.

    dc_ : array [n_sample]
        The radius of the optimal neighborhood for each point.


    Examples
    --------
    >>> from DPA import PAk
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> est = PAk.PointAdaptive_kNN().fit(X)
    >>> est.distances_
    array([[0.        , 1.        , 1.        ],
           [0.        , 1.        , 1.41421356],
           [0.        , 1.        , 1.41421356],
           [0.        , 1.41421356, 2.23606798],
           [0.        , 1.        , 2.23606798],
           [0.        , 1.        , 1.41421356]])
    >>> est.indices_
    array([[0, 1, 2],
           [1, 0, 2],
           [2, 0, 1],
           [3, 5, 4],
           [4, 5, 3],
           [5, 4, 3]])    
    
    References
    ----------
    # TODO
        
    """
    def __init__(self, k_max=1000, D_thr=23.92812698, metric="euclidean", dim_algo="auto", 
                       blockAn=True, block_ratio=20, frac=1, dim=None, n_jobs=None):
        self.k_max = k_max
        self.D_thr = D_thr
        self.metric = metric
        self.dim_algo = dim_algo
        self.blockAn = blockAn
        self.block_ratio = block_ratio
        self.frac = frac
        self.dim = dim
        self.n_jobs = n_jobs

        if metric not in VALID_METRIC:
            raise ValueError("invalid metric: '{0}'".format(metric))

        if dim_algo not in VALID_DIM:
            raise ValueError("invalid dim_algo: '{0}'".format(dim_algo))

        if self.dim_algo == "twoNN" and self.frac > 1:
            raise ValueError("frac should be between 0 and 1.")


    def fit(self, X, y=None):
        """Fit the PAk estimator on the data.

        Parameters
        ----------
        X : array [n_samples, n_samples] if metric == “precomputed”, or, 
            [n_samples, n_features] otherwise
            The input samples.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = check_array(X, order='C', accept_sparse=True)

        if not self.dim:
            if self.dim_algo == "auto":
                self.dim = X.shape[1]
            elif self.dim_algo == "twoNN":
                if self.block_ratio >= X.shape[0]:
                    raise ValueError("block_ratio is larger than the sample size, the minimum size for block analysis \
                                would be zero. Please set a lower value.")
                self.dim = twoNearestNeighbors(blockAn=self.blockAn, block_ratio=self.block_ratio, 
                                               frac=self.frac, n_jobs=self.n_jobs).fit(X).dim_
            else:
                pass

        if self.k_max > X.shape[0]:
            self.k_max = int(X.shape[0]/2)
        if self.k_max < 3:
            raise ValueError("k_max is below 3, the minimum value required for \
                        statistical significance. Please use a larger datasets.")

        if self.metric == "precomputed":
            nbrs = NearestNeighbors(n_neighbors=self.k_max+1, # The point i is counted in its neighborhood 
                                          algorithm="brute", 
                                        metric=self.metric,
                                        n_jobs=self.n_jobs).fit(X)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k_max+1, # The point i is counted in its neighborhood 
                                         algorithm="auto", 
                                        metric=self.metric, 
                                        n_jobs=self.n_jobs).fit(X)
        self.distances_, self.indices_ = nbrs.kneighbors(X) 

        self.densities_, self.err_densities_, self.k_hat_, self.dc_ = _PointAdaptive_kNN(self.distances_, 
                                                                                 self.indices_,
                                                                              k_max=self.k_max, 
                                                                              D_thr=self.D_thr, 
                                                                                  dim=self.dim)
        self.is_fitted_ = True

        return self



