from .ml_cluster_base import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

"""
density based clustering algorithms
"""

class DBCluster(ClusterBase):
    _pairwise = True
    
    def __init__(self, trainer):
        self.trainer = trainer
        # cluster labels
        self.labels = None
        # number of clusters
        self.n_clusters = None
        # number of noise points
        self.n_noise = None
    
    def fit(self,dmatrix,rho=None):

        '''fit the clustering model, assume input of NxN distance matrix or Nxm coordinates'''
        self.labels = self.trainer.fit(dmatrix, rho = None)
        # Number of clusters in labels, ignoring noise if present.
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = list(self.labels).count(-1)
        print('Estimated number of clusters: %d' % self.n_clusters)
        print('Estimated number of noise points: %d' % self.n_noise)

    def get_cluster_labels(self,index=[]):
        '''return the label of the samples in the list of index'''
        if (len(index) == 0): 
            return self.labels
        else:
            return self.labels[index]
    def get_n_cluster(self):
        return self.n_clusters
    def get_n_noise(self):
        return self.n_noise
    def pack(self):
        '''return all the info'''
        state = dict(trainer=self.trainer, trainer_params=self.trainer.pack(),
                 labels=self.labels, n_clusters=self.n_clusters, n_noise=self.n_noise)
        return state


class sklearn_DB(FitClusterBase):

    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        This is not a maximum bound on the distances of points within a cluster. 
        This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.

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
        self.metric = metrictype # e.g. 'euclidean'
        # distance metric
        self.eps = eps
        # The number of samples in a neighborhood for a point to be considered as a core point.
        self.min_samples = min_samples
        self.db = None

    def fit(self, dmatrix, rho = None):
        self.db = DBSCAN(eps=self.eps, min_samples=self.min_samples,metric=self.metric).fit(dmatrix)
        return self.db.labels_

    def pack(self):
        '''return all the info'''
        return self.db.get_params


class LAIO_DB(FitClusterBase):
    """Laio Clustering scheme
    $$ \rho_i\,=\,\sum_j{\chi(d_{ij}-d_{cut})}$$
    $$ \delta_i\,=\,\min_{j:\rho_j>\rho_i}(d_{ij})$$
    """

    _pairwise = True
    
    def __init__(self, deltamin):
        self.deltamin = deltamin

    def fit(self,dmatrix,rcut, min_samples):

        rho = self.estimate_rho(rcut,dmatrix)
        delta,nneigh = self.estimate_delta(rho,dmatrix)

        ###
        plt.scatter(rho,delta)
        plt.plot([min(rho),max(rho)],[dlim,dlim],c='red')
        plt.show()
        ###
        nclust = 0
        cl = np.zeros(len(rho),dtype='int')-1
        for i in range(len(rho)):
            if (rho[i] > min_samples and delta[i] > self.deltamin):
                nclust += 1
                cl[i] = nclust

        ### Assignment
        ordrho = np.argsort(rho)[::-1]
        rho_ord = rho[ordrho]
        for i in range(len(rho)):
            if cl[ordrho[i]]==-1:
                cl[ordrho[i]]=cl[nneigh[ordrho[i]]]

        return cl

    def estimate_delta(rho,dist):
        delta = (rho*0.0).copy()
        nneigh = np.ones(len(delta),dtype='int')
        for i in range(len(rho)):
            js = np.where(rho>rho[i])[0]
            if len(js)==0:
                delta[i] = np.max(dist[i,:])
                nneigh[i] = i
            else:
                delta[i] = np.min(dist[i,js])
                nneigh[i] = js[np.argmin(dist[i,js])]
        return delta, nneigh

    def estimate_rho(dcut,dist):
        ### Using a Gaussian Kernel, like in Laio's
        rho = np.zeros(len(dist))
        for i in range(len(dist)):
            for j in range(len(dist)):
                if(dist[i,j]<dcut): 
                    rho[i]+= np.exp(-(dist[i,j]/dcut)*(dist[i,j]/dcut))
        return rho

