"""
Density-based Clustering Algorithms
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import json
from yaml import dump as ydump
from yaml import Dumper
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .ml_cluster_base import ClusterBase, FitClusterBase
from ..io import NpEncoder

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
        state = dict(trainer=self.trainer.name, trainer_params=self.trainer.pack(), labels=self.labels.tolist(),
                     n_clusters=self.n_clusters, n_noise=self.n_noise)
        return state

    def save_state(self, filename, mode='json'):
         if mode == 'yaml':
             with open(filename+'-clustering-state.yaml', 'w') as yd:
                 ydump(self.pack(), yd, sort_keys=True, Dumper=Dumper)
         else:
             with open(filename+'-clustering-state.json', 'w') as jd:
                 json.dump(self.pack(), jd, sort_keys=True, cls=NpEncoder)

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
        self.name='DBSCAN'
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
        return self.db.get_params()


class LAIO_DB(FitClusterBase):
    """
    Clustering by fast search and find of density peaks, Rodriguez and Laio (2014).

    https://science.sciencemag.org/content/sci/344/6191/1492.full.pdf

    :math: \rho_i\,=\,\sum_j{\chi(d_{ij}-d_{cut})}
    :math: \delta_i\,=\,\min_{j:\rho_j>\rho_i}(d_{ij})
    """
    def __init__(self, distances=None, indices=None, dens_type=None, dc=None, percent=2.0):

        """
        Parameters
        ----------
        distances: numpy array 
        array of distances between data points and their neighbours of shape (Nele, n_neigh_comp)
        where Nele is the number of data points and n_neigh_comp is the number of neighbours to consider.

        indices: numpy array 
               array of shape (Nele, n_neigh_comp) where row i gives the indices of the neighbours for data point i

        dens_type: str
                 The type of density to compute. Can be 'exp' (exponential) or None (linear)

        dc: float 
        giving the cutoff distance beyond which data points don't contribute to the local density computation
        of another datapoint.

        percent: int
        Criterion for choosing dc, int from 0-100, typically chosen such that the average number of neighbours
        is 1-2% of the total number of points in the dataset. Default is 2%.
        """

        self.name='LAIO_DB'
        self.distances = distances
        self.indices = indices
        self.dens_type = dens_type
        self.dc = dc
        self.percent = percent

        # numpy array of shape (Nele,) where the densities of the data points expressed in terms of the number of data
        # points within the cutoff distance
        self.dens = None

        # numpy array of shape (Nele,) of the distances to the nearest data point to i
        # possessing a higher local density.
        self.delta_to_neigh = None

        # numpy array of shape (Nele,) where the ith entry gives the index of the nearest neighbour with
        # higher density than data point i.
        self.ref_neigh = None

        # Unused currently
        self.decision_graph = None

        # Clusters must lie at a distance greater than self.delta_cut (float) apart to be designated as independent
        # clusters
        self.delta_cut = None

        # If a "cluster" has a density smaller than self.dens_cut (float) it is discarded as noise
        self.dens_cut = None

        # numpy array of shape (Nele, ) giving the cluster (int from 0 to N) each data point belongs to.
        # Clusters are labelled by density. Cluster 0 has the highest density and cluster N has the smallest density.
        self.cluster = None

        # numpy array of shape (N,) giving the indices of the N cluster centres.
        self.center_indices = None

        # numpy array of shape (Nele,) where halo points are designated as -1 and otherwise are assigned to their
        # respective cluster centres
        self.halo = None

    def get_dc(self, data):
        """
        Compute the cutoff distance given the data.

        Parameters
        ----------
        data: np.matrix 
        The data array of shape (Nele, proj_dim) where N is the number of data points and proj_dim is the number of
        projected components of the kernel matrix.

        Returns
        -------
        self.dc: float
             the cutoff distance
        """

        Nele = data.shape[0]

        # The number of neighbours to consider is dictated by the percent parameter provided (default is 2% of the
        # total data points.
        n_neigh = int(self.percent/100.*Nele)

        # Compute with 4 times the recommended percentage of neighbours in order that we may re-estimate the percent
        # value supplied (rper) based on the data.
        n_neigh_comp = int(4.*self.percent/100.*Nele)
        neigh = NearestNeighbors(n_neighbors=n_neigh_comp).fit(data)
        self.distances, self.indices = neigh.kneighbors(data)

        # The cutoff distance dc is the average distance (averaged over all data points) of the "0.02*Nele"th neighbour
        dc = np.average(self.distances[:, n_neigh])

        # To store the local densities of each data point
        dens = np.zeros(data.shape[0])
        tt = 1.0
        factor = 1.0

        # We update dc based on a re-estimated percentage rper. Stop when rper is close to self.percent
        while tt > 0.05:
            dc = dc*factor
            for i in range(Nele):
                a = self.distances[i, :]
                dens[i] = len(a[(a <= dc)])
            rper = 100.*np.average(dens)/Nele
            tt = rper/self.percent - 1.
            if tt > 0.:
                factor = factor/1.001
            else:
                factor = factor*1.001
            tt = abs(tt)
        self.dc = dc

    def get_decision_graph(self, data, fplot=True):
        """
        Method currently doesn't produce the decision graph.

        Parameters
        ----------
        data: numpy array of shape (Nele, proj_dim).
        fplot: Boolean indicating whether or not to plot the decision graph

        Returns
        -------

        """

        Nele = data.shape[0]
        self.dens = np.zeros(Nele)

        if self.dc == None:
            self.get_dc(data)

        else:  # check compatibility between the provided cutoff distance and the provided distance matrix
            n_neigh_comp = int(10.*self.percent/100.*Nele)
            neigh = NearestNeighbors(n_neighbors=n_neigh_comp).fit(data)
            self.distances, self.indices = neigh.kneighbors(data)
            if self.dc > np.min(self.distances[:, n_neigh_comp - 1]):
                print("dc too big for being included within the", 10.*self.percent, "%  of data, consider using a "
                                                                                    "small dc or augment the percent "
                                                                                    "parameter")

        for i in range(Nele):
            a = self.distances[i, :]
            self.dens[i] = len(a[(a <= self.dc)])

        if self.dens_type == 'exp':  # Density is no longer number but exponential
            for i in range(Nele):
                a = self.distances[i, :]/self.dc  # distances expressed as a fraction of the cutoff distance
                self.dens[i] = np.sum(np.exp(-a**2))

        # The densities computed by this class or not log densities. Log densities are returned by the KDE class.

        # self.dens_cut = 0.2 * np.mean(np.log(self.dens)) + 0.8 * np.min(np.log(self.dens))
        self.dens_cut = 0.2 * np.mean(self.dens) + 0.8 * np.min(self.dens)
        self.delta_cut = np.mean(self.distances)

        self.delta_to_neigh = np.zeros(data.shape[0])
        self.ref_neigh = np.zeros(data.shape[0], dtype='int')
        indices = np.arange(data.shape[0])
        imax = []  # holds index of data point with highest local density

        for i in range(data.shape[0]):
            higher_dens_indices = indices[((self.dens > self.dens[i]) & (indices != i))]
            higher_dens_data = data[((self.dens > self.dens[i]) & (indices != i))]
            if higher_dens_data.shape[0] > 0:
                ds = np.transpose(sp.spatial.distance.cdist([np.transpose(data[i, :])], higher_dens_data))
                j = np.argmin(ds)
                self.ref_neigh[i] = higher_dens_indices[j]
                self.delta_to_neigh[i] = ds[j]
            else:
                self.delta_to_neigh[i] = -100.
                imax.append(i)
        self.delta_to_neigh[imax] = np.max(self.delta_to_neigh)*1.05

        # Plot kernel density on x-axis and delta_to_neigh (point of higher density) on the y-axis

        if fplot:

            plt.scatter(self.dens, self.delta_to_neigh)
            plt.plot([min(self.dens), max(self.dens)], [self.delta_cut, self.delta_cut], c='red')
            plt.plot([self.dens_cut, self.dens_cut], [np.min(self.distances), np.max(self.distances)], c='red')
            plt.title('Decision Graph')
            plt.xlabel('rho')
            plt.ylabel('delta')
            plt.show()

    def get_assignation(self, data):
        """
        Parameters
        ----------
        data: numpy array of shape (Nele, proj_dim) where proj_dim gives the number of dimensions the kernel matrix is
              projected to
        Returns: self.halo numpy array of shape (Nele, ) where halo points are designated as -1 and otherwise are
        assigned to their respective cluster centres.
        """
        ordered_by_dens = np.argsort(-self.dens)  # data points in descending order of density
        self.cluster = np.zeros(data.shape[0], dtype='int')
        indices = np.arange(data.shape[0])
        center_label = np.zeros(data.shape[0], dtype='int')
        ncluster = -1
        for i in range(data.shape[0]):  # trick to iterate through the ordered_by_dens array
            j = ordered_by_dens[i]
            if (self.dens[j] > self.dens_cut) & (self.delta_to_neigh[j] > self.delta_cut):
                ncluster = ncluster + 1
                self.cluster[j] = ncluster
                center_label[j] = ncluster
            else:
                # the assigned cluster is the cluster the closest neighbour of higher density belongs to
                self.cluster[j] = self.cluster[self.ref_neigh[j]]
                center_label[j] = -1
        self.center_indices = indices[(center_label != -1)]
        # border points is a binary array, 1 if data point has a neighbour that has been assigned a different cluster
        # centre and 0 otherwise.
        border_points = np.zeros(data.shape[0], dtype='int')
        self.halo = np.copy(self.cluster)

        for i in range(data.shape[0]):
            for j in self.indices[i, :][(self.distances[i, :] <= self.dc)]:
                if self.cluster[i] != self.cluster[j]:
                    border_points[i] = 1
        halo_cutoff = np.zeros(ncluster + 1)
        for i in range(ncluster + 1):
            # Extract the densities of a cluster's border points
            dens_of_border_points = self.dens[((border_points == 1) & (self.cluster == i))]
            if len(dens_of_border_points) == 0:  # if a cluster has no border points then it has no halo points.
                halo_cutoff[i] = 0
            else:
                halo_cutoff[i] = np.max(dens_of_border_points)
            i += 1
        self.halo[indices[(self.dens < halo_cutoff[self.cluster])]] = -1

        return self.halo

    def fit(self, data, rho=None):
        """
        Compute the center labels.

        Parameters
        ----------
        data: numpy array of shape (Nele, proj_dim) where proj_dim is the number of components the kernel matrix has been
              projected to.
        rho: densities, default is None since the DP.py module computes the densities itself.
        Returns: cluster_labels: numpy array of shape (Nele,) giving the cluster (int from 0 to N) each data point
        belongs to. Halo points are designated as -1.
        -------
        """

        self.get_dc(data)
        self.get_decision_graph(data)
        cluster_labels = self.get_assignation(data)

        return cluster_labels

    def pack(self):
        """
        Return dictionary containing the cutoff distance, self.dc, for data points to contribute to the local density
        of another data point as well as self.dens_cut, the density threshold for defining a cluster.
        """
        state = dict(deltamin=self.dc, rhomin=self.dens_cut)
        return state


"""
Legacy Code below for old Fast Search and Find of Density Peaks class.
"""


class old_LAIO(FitClusterBase):
    """Laio Clustering scheme
    Clustering by fast search and find of density peaks
    https://science.sciencemag.org/content/sci/344/6191/1492.full.pdf

    :math: \\rho_i\,=\,\\sum_j{\\chi(d_{ij}-d_{cut})}
         i.e. the local density of data point x_i
    :math: \\delta_i\,=\,\\min_{j:\\rho_j>\\rho_i}(d_{ij})
         i.e. the minimum distance to a neighbour with higher density


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
        :param deltamin: the lower bound on the distance between two cluster centres
        :param rhomin: The lower bound in kernel density, if the density of a cluster is lower
                       than this threshold, this "cluster" will be discarded as noise
        """

        self.deltamin = deltamin
        self.rhomin = rhomin

    def fit(self, dmatrix, rho=None):

        """

        Parameters
        ----------
        dmatrix: The distance matrix of shape (Nele, Nele)
        rho: The log densities of the points of shape (Nele,)

        Returns
        -------

        """

        if rho is None:
            raise ValueError('for fdb it is better to compute kernel density first')

        delta, nneigh = self.estimate_delta(dmatrix, rho)

        #  if there's no input values for rhomin and deltamin, we use simple heuristics
        if self.rhomin < 0:
            self.rhomin = 0.2*np.mean(rho) + 0.8*np.min(rho)
        if self.deltamin < 0:
            self.deltamin = np.mean(delta)

        #here we make the decision graph
        #x axis (rho) is the kernel density for each data point
        #y axis (delta) is the distance to the nearest higher density points

        plt.scatter(rho, delta)
        plt.plot([min(rho), max(rho)], [self.deltamin, self.deltamin], c='red')
        plt.plot([self.rhomin, self.rhomin], [min(delta), max(delta)], c='red')
        plt.xlabel('rho')
        plt.ylabel('delta')
        plt.show()

        nclust = 0  # cluster index
        cl = np.zeros(len(rho), dtype='int')-1  # numpy array of -1's
        for i in range(len(rho)):
            if rho[i] > self.rhomin and delta[i] > self.deltamin:
                nclust += 1
                cl[i] = nclust

        # Assignment
        ordrho = np.argsort(rho)[::-1]  # Indices of data points in descending order of local density
        rho_ord = rho[ordrho]
        for i in range(len(rho)):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]
        return cl

    def estimate_delta(self, dist, rho):
        """
        For each data point i, compute the distance (delta_i) between i and j,
        j is the closest data point that has a density higher then i, i.e. rho(j) > rho(i).

        Parameters
        ----------
        dist: distance matrix of shape (Nele, Nele)
        rho: log densities for each data point array of shape (Nele,)

        Returns: delta: numpy array of distances to nearest cluster centre for each datapoint.
                 nneight: numpy array giving the index of the nearest cluster centre.
        -------

        """
        delta = (rho*0.0).copy()
        nneigh = np.ones(len(delta), dtype='int')
        for i in range(len(rho)):
            #  for data i, find all points that have higher density
            js = np.where(rho > rho[i])[0]
            #  if there's no j's that have higher density than i, we set delta_i to be a large distance
            if len(js) == 0:
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
