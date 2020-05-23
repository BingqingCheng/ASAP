"""
class and methods for performing kernel density estimation
"""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

class Kernel_Density_Base:
    """
    Base class for performing kernel density estimation
    """
    def __init__(self):
        self.acronym = 'none'

    def fit(self, X):

        """Fit kernel model to X"""

        pass

    def evaluate_density(self, X):
        """Given an array of data, computes the local density of every point using kernel density estimation

        Input
        ------
        Data X : array, shape(n_sample,n_feature)

        Return
        ------
        Log of densities for every point: array, shape(n_sample)
        """
        pass

    def fit_evaluate_density(self, X):
        self.fit(X)
        return self.evaluate_density(X)

    def get_acronym(self):
        # we use an acronym for each KDE, so it's easy to find it and refer to it
        return self.acronym

class KDE_scipy(Kernel_Density_Base):
    """
    Kernel Density Estimation with Scipy
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    """
    def __init__(self, bw_method=None):
        """
        bw_method: str, scalar or callable, optional

        The method used to calculate the estimator bandwidth. 
        This can be ‘scott’, ‘silverman’, a scalar constant or a callable. 
        If a scalar, this will be used directly as kde.factor. 
        If a callable, it should take a gaussian_kde instance as only parameter and return a scalar. 
        If None (default), ‘scott’ is used. 
        """
        self.bw_method = bw_method
        self.kde = None
        self.acronym = 'kde_scipy'
        self._fitted = False

    def fit(self, X):
        """
        X: dataset, array_like

        Datapoints to estimate from. In case of univariate data this is a 1-D array, 
        otherwise a 2-D array with shape (# of data, # of dimension)

        Note that scipy.stats.gaussian_kde take X with shape (# of dimension, # of data)
        This is why we transpose the input X.
        """
        if isinstance(self.bw_method, float):
            # Note that scipy weights its bandwidth by the covariance of the
            # input data.  To make the results comparable to the other methods,
            # we divide the bandwidth by the sample standard deviation here.
            self.kde = gaussian_kde(X.T, bw_method=self.bw_method/x.std(ddof=1))
        else:
            self.kde = gaussian_kde(X.T, bw_method=self.bw_method)
        self._fitted = True

    def evaluate_density(self, X):
        if self._fitted is False:
            raise ValueError("The KDE model has not been fitted.")
        return np.log(self.kde.evaluate(X.T))
        

class KDE_sklearn(Kernel_Density_Base):
    """
    Kernel Density Estimation with Sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
    https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation
    """
    def __init__(self, bandwidth=1.0, algorithm='auto', kernel='gaussian', metric='euclidean'):
        """
        bandwidth: float. The bandwidth of the kernel.
        algorithm: str. The tree algorithm to use. Valid options are [‘kd_tree’|’ball_tree’|’auto’]. Default is ‘auto’.
        kernel:str. The kernel to use. 
                    Valid kernels are [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’] 
                    Default is ‘gaussian’.
        metric: str. The distance metric to use. 
        """
        self.bandwidth = bandwidth
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.kde = KernelDensity(bandwidth=bandwidth, algorithm=algorithm, kernel=kernel, metric=metric)
        self.acronym = 'kde_sklearn'
        self._fitted = False

    def fit(self, X):
        """
        X: dataset, array_like

        Datapoints to estimate from. In case of univariate data this is a 1-D array,
        otherwise a 2-D array with shape (# of data, # of dimension)

        """
        self.kde.fit(X)
        self._fitted = True

    def evaluate_density(self, X):
        if self._fitted is False:
            raise ValueError("The KDE model has not been fitted.")
        return self.kde.score_samples(X)
