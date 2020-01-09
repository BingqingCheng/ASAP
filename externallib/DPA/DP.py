"""
Module implementing the "Fast Search and Find of Density Peaks" clustering algorithm of Rodriguez and Laio (2014).
"""

import numpy as np
import scipy as sp
from scipy import stats
from sklearn.neighbors import NearestNeighbors


class Density_Peaks_clustering:
    """
    Clustering by fast search and find of density peaks, Rodriguez and Laio (2014).
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
        self.delta_cut = None
        self.dens_cut = None
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
