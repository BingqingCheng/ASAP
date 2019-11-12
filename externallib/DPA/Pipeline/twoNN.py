# TWO-NN: Intrinsic dimension estimator by a minimal neighborhood information.
#
# Author: Maria d'Errico <mariaderr@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
import heapq
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, DensityMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from math import log, sqrt, exp, lgamma, pi, pow
from sklearn.linear_model import LinearRegression

def _twoNearestNeighbors(distances, blockAn=True, block_ratio=20, frac=1):
    """Main function for the TWO-NN estimator.

    Parameters
    ----------
    distances: array [n_samples, k_max]
        Distances to the k_max neighbors of each points.
 
    blockAn : bool, default=True
        If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant dimensions 
        as a function of the block size. This allows to study the stability of the estimation with respect to
        changes in the neighborhood size, which is crucial for ID estimations when the data lie on a manifold perturbed 
        by a high-dimensional noise. 

    block_ratio : int, default=20
        Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False, block_ratio is ignored.

    frac : float, default=1
        Define the fraction of points in the data set used for ID calculation. By default the full data set is used.   
    """
    N = len(distances)
    log_nu = []
    d_mle = 0.
    for i in range(0, N):
        # TODO: handle duplicate points, i.e. distances[i][1] equals to zero
        log_nu_i = log(distances[i][2])-log(distances[i][1])
        log_nu.append(log_nu_i) 
        d_mle = d_mle + log_nu_i
    # Intrinsic dimension estimated with MLE variant of TWO-NN
    id_mle = int(round(N/d_mle))

    # Perform linear fit -log[1-F(nu)]=d*log(nu), where F(nu)=i/N, with intercept=0
    x = log_nu
    if frac<1:
        x = []
        heapq.heapify(log_nu)
        for _ in range(int(round(len(log_nu)*frac))):
            x.append(heapq.heappop(log_nu))
    else:
        x = sorted(log_nu)

    # TODO: compare with the available benchmarks using (i+1)/N instead
    #y = [-log(1.-(i+1)/N) for i in range(len(x))]    
    y = [-log(1.-(i)/N) for i in range(len(x))]    
    # Fit the model y = a * x using np.linalg.lstsq
    a, _, _, _ = np.linalg.lstsq(np.array(x)[:,np.newaxis], np.array(y), rcond=None)
    id = a[0]
    
    # TODO: Implement block analysis and decimation plot. It doesn't affect the results.

    return int(round(id)), x, y



class twoNearestNeighbors(BaseEstimator, DensityMixin):
    """Class definition for TWO-NN: Intrinsic dimension estimator by a minimal neighborhood information.

    The TWO-NN estimator uses only the distances to the first two nearest neighbors of each point.


    Parameters
    ----------
    metric : string, or callable
        The distance metric to use. 
        If metric is a string, it must be one of the options allowed by 
        scipy.spatial.distance.pdist for its metric parameter, or a metric listed in 
        pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is "precomputed", X is assumed to
        be a distance matrix. Alternatively, if metric is a callable function, it is 
        called on each pair of instances (rows) and the resulting value recorded. The 
        callable should take two arrays from X as input and return a value indicating 
        the distance between them. Default is 'euclidean'.

    blockAn : bool, default=True
        If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant dimensions 
        as a function of the block size. This allows to study the stability of the estimation with respect to
        changes in the neighborhood size, which is crucial for ID estimations when the data lie on a manifold perturbed 
        by a high-dimensional noise. 

    block_ratio : int, default=20
        Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False, block_ratio is ignored.

    frac : float, default=1
        Define the fraction of points in the data set used for ID calculation. By default the full data set is used.   

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    dim_ = int
        The intrinsic dimensionality
 
    References
    ----------
    # TODO


    """
    def __init__(self, metric='euclidean',  blockAn=True, block_ratio=20, frac=1, n_jobs=None):
        self.metric = 'euclidean'
        self.blockAn = blockAn
        self.block_ratio = block_ratio
        self.frac = frac
        self.n_jobs = n_jobs

        if self.frac > 1:
            raise ValueError("frac should be between 0 and 1.")
    
    
    def fit(self, X, y=None):
        """A reference implementation of a fitting function.

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

        if self.block_ratio >= X.shape[0]:
            raise ValueError("block_ratio is larger than the sample size, the minimum size for block analysis \
                        would be zero. Please set a lower value.")

        if self.metric == "precomputed":
            # TODO: handle identical distances
            nbrs = NearestNeighbors(n_neighbors=3, # Only two neighbors used; the point i is counted in its neighborhood
                                          algorithm="brute",
                                        metric=self.metric,
                                        n_jobs=self.n_jobs).fit(X)
        else:
            # Remove duplicates coordinates
            X = np.unique(X, axis=0)
            nbrs = NearestNeighbors(n_neighbors=3, # Only two neighbors used; the point i is counted in its neighborhood
                                         algorithm="auto",
                                        metric=self.metric, 
                                        n_jobs=self.n_jobs).fit(X)
        self.distances_, self.indices_ = nbrs.kneighbors(X) 

     
        self.dim_, self.x_, self.y_ = _twoNearestNeighbors(self.distances_, 
                                    blockAn=self.blockAn, 
                             block_ratio=self.block_ratio, 
                                          frac=self.frac)
        self.is_fitted_ = True
        return self


