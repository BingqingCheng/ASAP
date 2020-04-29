"""
TODO: Module-level desription
I adapted the code from:
https://github.com/alexandreday/fast_density_clustering.git
Copyright 2017 Alexandre Day
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors


class KDE:
    """Kernel density estimation (KDE) for accurate local density estimation.
    This is achieved by using maximum-likelihood estimation of the generative kernel density model
    which is regularized using cross-validation.


    Parameters
    ----------
    bandwidth: float, optional
        bandwidth for the kernel density estimation. If not specified, will be determined automatically using
        maximum likelihood on a test-set.

    nh_size : int, optional (default = 'auto')
        number of points in a typical neighborhood... only relevant for evaluating
        a crude estimate of the bandwidth.'auto' means that the nh_size is scaled with number of samples. We 
        use nh_size = 100 for 10000 samples. The minimum neighborhood size is set to 4.

    test_ratio_size: float, optional (default = 0.8)
        Ratio size of the test set used when performing maximum likehood estimation.
        In order to have smooth density estimations (prevent overfitting), it is recommended to
        use a large test_ratio_size (closer to 1.0) rather than a small one.

    atol: float, optional (default = 0.000005)
        kernel density estimate precision parameter. determines the precision used for kde.
        smaller values leads to slower execution but better precision
    
    rtol: float, optional (default = 0.00005)
        kernel density estimate precision parameter. determines the precision used for kde.
        smaller values leads to slower execution but better precision
    
    xtol: float, optional (default = 0.01)
        precision parameter for optimizing the bandwidth using maximum likelihood on a test set

    test_ratio_size: float, optional
        ratio of the test size for determining the bandwidth.

    kernel: str, optional (default='gaussian')
        Type of Kernel to use for density estimates. Other options are {'epanechnikov'|'linear','tophat'}.
    """

    def __init__(self, nh_size='auto', bandwidth=None, test_ratio_size=0.1,
                 xtol=0.01, atol=0.000005, rtol=0.00005, extreme_dist=False,
                 nn_dist=None, kernel='gaussian'):

        self.bandwidth = bandwidth
        self.nh_size = nh_size
        self.test_ratio_size = test_ratio_size
        self.xtol = xtol
        self.atol = atol
        self.rtol = rtol
        self.extreme_dist = extreme_dist
        self.nn_dist = nn_dist
        self.kernel = kernel  # epanechnikov other option

    def fit(self, X):

        """Fit kernel model to X"""

        if self.nh_size is 'auto':
            self.nh_size = max([int(25 * np.log10(X.shape[0])), 4])

        if X.shape[1] > 8:
            print('Careful, you are trying to do density estimation for data in a D > 8 dimensional space\n'
                  ' ... you are warned !')

        if self.bandwidth is None:
            self.bandwidth = self.find_optimal_bandwidth(X)
        else:
            self.kde = KernelDensity(
                bandwidth=self.bandwidth, algorithm='kd_tree',
                kernel=self.kernel, metric='euclidean',
                atol=self.atol, rtol=self.rtol,
                breadth_first=True, leaf_size=40
            )
        self.kde.fit(X)
        return self

    def evaluate_density(self, X):
        """Given an array of data, computes the local density of every point using kernel density estimation

        Input
        ------
        Data X : array, shape(n_sample,n_feature)

        Return
        ------
        Log of densities for every point: array, shape(n_sample)
        Return:
            kde.score_samples(X)
        """
        return self.kde.score_samples(X)

    def bandwidth_estimate(self, X_train, X_test):
        """Gives a rough estimate of the optimal bandwidth (based on the notion of some effective neigborhood)
        
        Return
        ---------
        bandwidth estimate, minimum possible value : tuple, shape(2)
        """

        if self.nn_dist is None:
            nn = NearestNeighbors(n_neighbors=self.nh_size, algorithm='kd_tree')
            nn.fit(X_train)
            nn_dist, _ = nn.kneighbors(X_test, n_neighbors=self.nh_size, return_distance=True)
        else:
            nn_dist = self.nn_dist

        dim = X_train.shape[1]

        # Computation of minimum bound
        # This can be computed by taking the limit h -> 0 and making a saddle-point approx.
        mean_nn2_dist = np.mean(nn_dist[:, 1] * nn_dist[:, 1])
        h_min = np.sqrt(mean_nn2_dist / dim)

        idx_1 = np.random.choice(np.arange(len(X_train)), size=min([1000, len(X_train)]), replace=False)
        idx_2 = np.random.choice(np.arange(len(X_test)), size=min([1000, len(X_test)]), replace=False)

        max_size = min([len(idx_1), len(idx_2)])

        tmp = np.linalg.norm(X_train[idx_1[:max_size]] - X_test[idx_2[:max_size]], axis=1)

        h_max = np.sqrt(np.mean(tmp * tmp) / dim)
        h_est = 10 * h_min
        return h_est, h_min, h_max

    def find_optimal_bandwidth(self, X):
        """Performs maximum likelihood estimation on a test set of the density model fitted on a training set
        """
        from scipy.optimize import fminbound
        X_train, X_test = train_test_split(X, test_size=self.test_ratio_size)
        args = (X_test,)

        hest, hmin, hmax = self.bandwidth_estimate(X_train, X_test)

        print("[kde] Minimum bound = %.4f \t Rough estimate of h = %.4f \t Maximum bound = %.4f" % (hmin, hest, hmax))

        # We are trying to find reasonable tight bounds (hmin, 4.0*hest) to bracket the error function minima
        # Would be nice to have some hard accurate bounds
        self.xtol = round_float(hmin)

        print('[kde] Bandwidth tolerance (xtol) set to precision of minimum bound : %.5f ' % self.xtol)

        self.kde = KernelDensity(algorithm='kd_tree', atol=self.atol, rtol=self.rtol, leaf_size=40, kernel=self.kernel)

        self.kde.fit(X_train)

        # hmax is the upper bound, however, heuristically it appears to always be way above the actual bandwidth. hmax*0.2 seems much better but still conservative
        h_optimal, score_opt, _, niter = fminbound(self.log_likelihood_test_set, hmin, hmax * 0.2, args, maxfun=100,
                                                   xtol=self.xtol, full_output=True)

        print("[kde] Found log-likelihood maximum in %i evaluations, h = %.5f" % (niter, h_optimal))

        if self.extreme_dist is False:  # These bounds should always be satisfied ...
            assert abs(h_optimal - hmax) > 1e-4, "Upper boundary reached for bandwidth"
            assert abs(h_optimal - hmin) > 1e-4, "Lower boundary reached for bandwidth"

        return h_optimal

    # @profile
    def log_likelihood_test_set(self, bandwidth, X_test):
        """Fit the kde model on the training set given some bandwidth and evaluates the negative log-likelihood of the test set
        """
        self.kde.bandwidth = bandwidth
        # l_test = len(X_test)
        return -self.kde.score(X_test[
                               :2000])  # X_test[np.random.choice(np.arange(0, l_test), size=min([int(0.5*l_test), 1000]), replace=False)]) # this should be accurate enough !


def round_float(x):
    """Rounds a float to it's first significant digit"""
    a = list(str(x))
    for i, e in enumerate(a):
        if e != '.':
            if e != '0':
                pos = i
                a[i] = '1'
                break
    return float("".join(a[:pos + 1]))
