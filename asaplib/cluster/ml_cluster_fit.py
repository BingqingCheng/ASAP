"""
Density-based clustering algorithms
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from .ml_cluster_base import ClusterBase, FitClusterBase


class DBCluster(ClusterBase):
    """
    Performing clustering using density based clustering algorithm
    """
    _pairwise = True

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        # cluster labels
        self.labels = None
        # number of clusters
        self.n_clusters = None
        # number of noise points
        self.n_noise = None

    def fit(self, dmatrix, rho=None):
        """fit the clustering model, assume input of NxN distance matrix or Nxm coordinates"""

        self.labels = self.trainer.fit(dmatrix, rho)
        # Number of clusters in labels, ignoring noise if present.
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = list(self.labels).count(-1)
        print('Estimated number of clusters: %d' % self.n_clusters)
        print('Estimated number of noise points: %d' % self.n_noise)

        if np.shape(dmatrix)[0] == np.shape(dmatrix)[1]:
            silscore = silhouette_score(dmatrix, self.labels, metric="precomputed")
        else:
            silscore = silhouette_score(dmatrix, self.labels, metric="euclidean")
        print("Silhouette Coefficient: %0.3f" % silscore)

    def get_cluster_labels(self, index=[]):
        """return the label of the samples in the list of index"""
        if len(index) == 0:
            return self.labels
        else:
            return self.labels[index]

    def get_n_cluster(self):
        return self.n_clusters

    def get_n_noise(self):
        return self.n_noise

    def pack(self):
        """return all the info"""
        state = dict(trainer=self.trainer, trainer_params=self.trainer.pack(), labels=self.labels,
                     n_clusters=self.n_clusters, n_noise=self.n_noise)
        return state


class sklearn_DB(FitClusterBase):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster.
        This is the most important DBSCAN parameter to choose appropriately for your dataset and distance function.

    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        This includes the point itself.

    metric : string, or callable
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter.
        If metric is “precomputed”, X is assumed to be a distance matrix and must be square.
        X may be a sparse matrix, in which case only “nonzero” elements may be considered neighbors for DBSCAN.
    """
    _pairwise = True

    def __init__(self, eps=None, min_samples=None, metrictype='precomputed'):
        super().__init__()
        self.metric = metrictype  # e.g. 'euclidean'
        # distance metric
        self.eps = eps
        # The number of samples in a neighborhood for a point to be considered as a core point.
        self.min_samples = min_samples
        self.db = None

    def fit(self, dmatrix, rho=None):
        self.db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit(dmatrix)
        return self.db.labels_

    def pack(self):
        """return all the info"""
        return self.db.get_params


class LAIO_DB(FitClusterBase):
    """Laio Clustering scheme
    Clustering by fast search and find of density peaks
    https://science.sciencemag.org/content/sci/344/6191/1492.full.pdf

    $$ \rho_i\,=\,\sum_j{\chi(d_{ij}-d_{cut})}$$
    $$ \delta_i\,=\,\min_{j:\rho_j>\rho_i}(d_{ij})$$

    A summary of laio clustering algorithm:
    1. First do a kernel density estimation (rho_i) for each data point i 
    2. For each data point i, compute the distance (delta_i) between i and j, 
       j is the closet data point that has a density higher then i, i.e. rho(j) > rho(i).
    3. Plot the decision graph, which is a scatter plot of (rho_i, delta_i)
    4. Select cluster centers ({cl}), which are the outliers in the decision graph that fulfills:
       i) rho({cl}) > rhomin
       ii) delta({cl}) > delta_min
       one needs to set the two parameters rhomin and delta_min.
    5. After the cluster centers are determined, data points are assigned to the nearest cluster center.
 one needs to set two parameters:
    """

    _pairwise = True

    def __init__(self, deltamin=-1, rhomin=-1):
        """
        rhomin: the lower bound in kernel density, if the density of a cluster center is 
                lower than this threshold, this "cluster" will be discarded as noise
        deltamin: the lower bound on the distance of two cluster center.
        """
        self.deltamin = deltamin
        self.rhomin = rhomin

    def fit(self, dmatrix, rho=None):

        """

        Parameters
        ----------
        dmatrix: The distance matrix
        rho: The log densities of the points

        Returns
        -------
        cluster label
        """

        if rho is None:
            raise ValueError('for fdb it is better to compute kernel density first')

        delta, nneigh = self.estimate_delta(dmatrix, rho)

        # if there's no input values for rhomin and deltamin, we use simple heuristics
        if self.rhomin < 0:
            self.rhomin = 0.2 * np.mean(rho) + 0.8 * np.min(rho)
        if self.deltamin < 0:
            self.deltamin = np.mean(delta)

        """
        here we make the decision graph
        x axis (rho) is the kernel density for each data point
        y axis (delta) is the distance to the nearest higher density points
        """
        plt.scatter(rho, delta)
        plt.plot([min(rho), max(rho)], [self.deltamin, self.deltamin], c='red')
        plt.plot([self.rhomin, self.rhomin], [min(delta), max(delta)], c='red')
        plt.xlabel('rho')
        plt.ylabel('delta')
        plt.show()
        ###

        nclust = 0
        cl = np.zeros(len(rho), dtype='int') - 1
        for i in range(len(rho)):
            # we select cluster centers {cl} with rho({c}) > rhomin && delta({c}) > delta_min
            if rho[i] > self.rhomin and delta[i] > self.deltamin:
                nclust += 1
                cl[i] = nclust

        ### Assignment
        ordrho = np.argsort(rho)[::-1]
        rho_ord = rho[ordrho]
        for i in range(len(rho)):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]
        return cl

    def estimate_delta(self, dist, rho):

        ''' For each data point i, compute the distance (delta_i) between i and j,
           j is the closet data point that has a density higher then i, i.e. rho(j) > rho(i).
        '''

    delta = (rho * 0.0).copy()
    nneigh = np.ones(len(delta), dtype='int')
    for i in range(len(rho)):
        # for data i, find all points that have higher density
        js = np.where(rho > rho[i])[0]
        if len(js) == 0:
            # if there's no j's that have higher density than i, we set delta_i to be a large distance
            delta[i] = np.max(dist[i, :])
            nneigh[i] = i
        else:
            # find the nearest j that has higher density then i
            delta[i] = np.min(dist[i, js])
            nneigh[i] = js[np.argmin(dist[i, js])]
    return delta, nneigh


def pack(self):
    '''return all the info'''
    state = dict(deltamin=self.deltamin, rhomin=self.rhomin)
    return state
