"""
Density-based Clustering Algorithms
"""

import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .ml_cluster_base import ClusterBase, FitClusterBase


class DBCluster(ClusterBase):
    """
    TODO: Make class-level docstring
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
        print("Silhouette Coefficient: %0.3f" %silscore)

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


# class LAIO_DB(FitClusterBase):
#     """Laio Clustering scheme
#     Clustering by fast search and find of density peaks
#     https://science.sciencemag.org/content/sci/344/6191/1492.full.pdf
#
#     $$ \rho_i\,=\,\sum_j{\chi(d_{ij}-d_{cut})}$$ i.e. the local density of data point x_i
#     $$ \delta_i\,=\,\min_{j:\rho_j>\rho_i}(d_{ij})$$ i.e. the minimum distance to a neighbour with higher density
#     """
#
#     _pairwise = True
#
#     def __init__(self, deltamin=-1, rhomin=-1):
#         self.deltamin = deltamin
#         self.rhomin = rhomin
#
#     def fit(self, dmatrix, rho=None):
#
#         """
#
#         Parameters
#         ----------
#         dmatrix: The distance matrix of shape (Nele, Nele)
#         rho: The log densities of the points of shape (Nele,)
#
#         Returns
#         -------
#
#         """
#
#         if rho is None:
#             raise ValueError('for fdb it is better to compute kernel density first')
#
#         delta, nneigh = self.estimate_delta(dmatrix, rho)
#
#         # I'll think about this!!! Criterion for deciding which data points act as cluster centres
#         if self.rhomin < 0:
#             self.rhomin = 0.2*np.mean(rho) + 0.8*np.min(rho)
#         if self.deltamin < 0:
#             self.deltamin = np.mean(delta)
#
#         plt.scatter(rho, delta)
#         plt.plot([min(rho), max(rho)], [self.deltamin, self.deltamin], c='red')
#         plt.plot([self.rhomin, self.rhomin], [min(delta), max(delta)], c='red')
#         plt.xlabel('rho')
#         plt.ylabel('delta')
#         plt.show()
#
#         nclust = 0  # cluster index
#         cl = np.zeros(len(rho), dtype='int')-1  # numpy array of -1's
#         for i in range(len(rho)):
#             if rho[i] > self.rhomin and delta[i] > self.deltamin:
#                 nclust += 1
#                 cl[i] = nclust
#
#         # Assignment
#         ordrho = np.argsort(rho)[::-1]  # Indices of data points in descending order of local density
#         rho_ord = rho[ordrho]
#         for i in range(len(rho)):
#             if cl[ordrho[i]] == -1:
#                 cl[ordrho[i]] = cl[nneigh[ordrho[i]]]
#         return cl
#
#     def estimate_delta(self, dist, rho):
#         """
#
#         Parameters
#         ----------
#         dist: distance matrix of shape (Nele, Nele)
#         rho: log densities for each data point array of shape (Nele,)
#
#         Returns: delta: numpy array of distances to nearest cluster centre for each datapoint.
#                  nneight: numpy array giving the index of the nearest cluster centre.
#         -------
#
#         """
#         delta = (rho*0.0).copy()
#         nneigh = np.ones(len(delta), dtype='int')
#         for i in range(len(rho)):
#             js = np.where(rho > rho[i])[0]
#             if len(js) == 0:
#                 delta[i] = np.max(dist[i, :])
#                 nneigh[i] = i
#             else:
#                 delta[i] = np.min(dist[i, js])
#                 nneigh[i] = js[np.argmin(dist[i, js])]
#         return delta, nneigh
#
#     def pack(self):
#         '''return all the info'''
#         state = dict(deltamin=self.deltamin, rhomin=self.rhomin)
#         return state


class LAIO_DB(FitClusterBase):
    """
    Clustering by fast search and find of density peaks, Rodriguez and Laio (2014).
    https://science.sciencemag.org/content/sci/344/6191/1492.full.pdf
    $$ \rho_i\,=\,\sum_j{\chi(d_{ij}-d_{cut})}$$
    $$ \delta_i\,=\,\min_{j:\rho_j>\rho_i}(d_{ij})$$
    """
    def __init__(self, distances=None, indices=None, dens_type="eps", dc=None, percent=2.0):

        """
        Parameters
        ----------
        distances: Matrix of distances between data points and their neighbours of dimension Nxn_neigh_comp where N is
        the number of data points and n_neigh_comp is the number of neighbours to consider.
        indices: The indices in the data matrix of the neighours for all of the data points.
        dens_type: The type of density to compute (can be exponential)
        dc: The cutoff distance beyond which data points don't contribute to the local density computation of another
        datapoint.
        percent: Criterion for choosing dc, int from 0-100, typically chosen such that the average number of neighbours is 1-2% of the
        total number of points in the dataset.
        """

        self.distances = distances
        self.indices = indices
        self.dens_type = dens_type
        self.dc = dc
        self.percent = percent
        self.dens = None  # numpy array of the densities of the data points
        self.delta = None  # numpy array of the distances to the nearest cluster centre
        self.ref = None  # numpy array of the indices of the cluster centres for each data point
        self.decision_graph = None
        self.delta_cut = None  # distance criterion for definining a cluster
        self.dens_cut = None  # density criterion for defining a cluster
        self.cluster = None
        self.centers = None
        self.halo = None

    def get_dc(self, data):

        """
        Compute the cutoff distance given the data.
        Parameters
        ----------
        data: The dataset of dimension mxn where m is the number of data points and n is the number of features.
        Returns
        -------
        self.dc, the cutoff distance
        """

        Nele = data.shape[0]  # The number of data points
        n_neigh = int(self.percent/100.*Nele)
        n_neigh_comp = int(4.*self.percent/100.*Nele)
        neigh = NearestNeighbors(n_neighbors=n_neigh_comp).fit(data)
        self.distances, self.indices = neigh.kneighbors(data)  # Distance matrix and indices are 4 times the length. Distance matrix shape is (Nele, n_neigh_comp)
        dc = np.average(self.distances[:, n_neigh])  # The cutoff distance is the average distance (averaged over all data points) of the "0.02*Nele"th neighbour
        dens = np.zeros(data.shape[0])  # Store the local densities of the data points in an array of shape (Nele,)
        tt = 1.0
        factor = 1.0
        while tt > 0.05:
            dc = dc*factor
            for i in range(Nele):
                a = self.distances[i, :]  # for a data point x_i, a gives the distances to all the n_neigh_comp neighbours.
                dens[i] = len(a[(a <= dc)])  # we count the number of data points within the cutoff distance from x_i
            rper = 100.*np.average(dens)/Nele
            tt = rper/self.percent - 1.
            if tt > 0.:
                factor = factor/1.001
            else:
                factor = factor*1.001
            tt = abs(tt)
        self.dc = dc

    def get_decision_graph(self, data):

        Nele = data.shape[0]
        self.dens = np.zeros(Nele)  # shape = (Nele,)

        if self.dc == None:
            self.get_dc(data)
        else:  # check compatibility between the provided cutoff distance and the provided distance matrix
            n_neigh_comp = int(10.*self.percent/100.*Nele)
            neigh = NearestNeighbors(n_neighbors=n_neigh_comp).fit(data)
            self.distances, self.indices = neigh.kneighbors(data)
            if self.dc > np.min(self.distances[:, n_neigh_comp - 1]):
                print("dc too big for being included within the", 10.*self.percent, "%  of data, consider using a small dc or augment the percent parameter")

        for i in range(Nele):
            a = self.distances[i, :]  # a are the distances of the n_neigh_comp neighbours relative to data points i
            self.dens[i] = len(a[(a <= self.dc)])  # The density of i is the number of points at a distance smaller than the cutoff

        if self.dens_type == 'exp':  # Density is no longer number but exponential
            for i in range(Nele):
                a = self.distances[i, :]/self.dc  # distances expressed as a fraction of the cutoff distance
                self.dens[i] = np.sum(np.exp(-a**2))

        # may need to re-think these criterion based on the type of density chosen

        self.dens_cut = 0.2 * np.mean(np.log(self.dens)) + 0.8 * np.min(np.log(self.dens))
        self.delta_cut = np.mean(self.distances)

        self.delta = np.zeros(data.shape[0])  # shape = (Nele,)
        self.ref = np.zeros(data.shape[0], dtype='int')  # shape = (Nele,)
        tt = np.arange(data.shape[0])  # array from 0-Nele
        imax = []
        for i in range(data.shape[0]):
            ll = tt[((self.dens > self.dens[i]) & (tt != i))]  # indices of data points with higher local density than x_i
            dd = data[((self.dens > self.dens[i]) & (tt != i))]  # data points with higher local density than x_i
            if dd.shape[0] > 0:  # If there is a data point with higher local density than x_i
                ds = np.transpose(sp.spatial.distance.cdist([np.transpose(data[i, :])], dd))
                j = np.argmin(ds)
                self.ref[i] = ll[j]  # cluster centre for data point x_i
                self.delta[i] = ds[j]  # distance to cluster centre for data point x_i
            else:
                self.delta[i] =- 100.
                imax.append(i)
        self.delta[imax] = np.max(self.delta)*1.05

    def get_assignation(self, data):
        """
        Parameters
        ----------
        data: data matrix
        Returns: center label: numpy array of shape (Nele,) giving the cluster label for a each data point.
        -------
        """
        ordered = np.argsort(-self.dens)  # in descending order of density
        self.cluster = np.zeros(data.shape[0], dtype='int')
        tt = np.arange(data.shape[0])
        center_label = np.zeros(data.shape[0], dtype='int')
        ncluster = -1
        for i in range(data.shape[0]):
            j = ordered[i]
            if (self.dens[j] > self.dens_cut) & (self.delta[j] > self.delta_cut):
                ncluster = ncluster + 1
                self.cluster[j] = ncluster
                center_label[j] = ncluster
            else:
                self.cluster[j] = self.cluster[self.ref[j]]
                center_label[j] = -1
        self.centers = tt[(center_label != -1)]
        bord = np.zeros(data.shape[0], dtype='int')
        self.halo = np.copy(self.cluster)

        for i in range(data.shape[0]):
            for j in self.indices[i, :][(self.distances[i, :] <= self.dc)]:
                if self.cluster[i] != self.cluster[j]:
                    bord[i] = 1
        halo_cutoff = np.zeros(ncluster + 1)
        for i in range(ncluster + 1):
            dd = self.dens[((bord == 1) & (self.cluster == i))]
            halo_cutoff[i] = np.max(dd)
        self.halo[tt[(self.dens < halo_cutoff[self.cluster])]] =- 1

        return center_label

    def fit(self, data, rho=None):
        """
        Compute the center labels.
        Parameters
        ----------
        data: data matrix of shpae (Nele, Nele)
        rho: densities, default is None since the DP.py module computes the densities itself.
        Returns: center_label: numpy array of shape (Nele,) giving the index of the cluster centre for each data point
        -------
        """

        self.get_dc(data)
        self.get_decision_graph(data)
        center_label = self.get_assignation(data)

        return center_label

    def pack(self):
        '''return all the info'''
        state = dict(deltamin=self.dc, rhomin=self.dens_cut)
        return state
