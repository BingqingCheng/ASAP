import numpy as np
from Pipeline import DPA
import hdbscan
import sklearn.cluster as cluster
from sklearn import mixture
from DP import Density_Peaks_clustering
import time
class example:
    def __init__(self, name):
        self.name = name
        self.file = 'Pipeline/tests/benchmarks/Fig'+name+'.dat'
        self.gt_file = 'Pipeline/tests/benchmarks/gt_F'+name+'.txt'
        self.data=np.loadtxt(self.file,dtype='float') #2D dataset
        self.gt=np.loadtxt(self.gt_file,dtype='int')
        self.dpa_Z=None
        self.hdbscan_min_cluster_size=None
        self.hdbscan_min_samples=None
        self.spect_n_clusters=None
        self.spect_n_neighbors=None
        self.dpa=None
        self.hdbscan=None
        self.hdbscan_labels=None
        self.spect=None
        self.spect_labels=None
        self.dpa_time=None
        self.hdbscan_time=None
        self.spect_time=None
        self.dpgmm_n_components=None
        self.dpgmm_random_state=None
        self.dpgmm_n_init=None
        self.dpgmm=None
        self.dpgmm_labels=None
        self.dpgmm_pop=None
        self.dpgmm_time=None
        self.dp=None
        self.dp_percent=None
        self.dp_dens_type=None
        self.dp_delta_cut=None
        self.dp_dens_cut=None
        self.dp_time=None
    def exe_dpgmm(self):
        start=time.time()
        self.dpgmm=mixture.BayesianGaussianMixture(n_components=self.dpgmm_n_components,covariance_type='full',random_state=self.dpgmm_random_state,n_init=self.dpgmm_n_init).fit(self.data)
        baypred=self.dpgmm.predict(self.data)
        tmp,self.dpgmm_labels,self.dpgmm_pop=np.unique(baypred, return_index=False, return_inverse=True, return_counts=True, axis=None)
        end=time.time()
        self.dpgmm_time=end-start
    def exe_dpa(self):
        self.dpa = DPA.DensityPeakAdvanced(Z=self.dpa_Z)
        start=time.time()
        self.dpa.fit(self.data)
        end=time.time()
        self.dpa_time=end-start
    def exe_hdbscan(self):
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=self.hdbscan_min_cluster_size,min_samples=self.hdbscan_min_samples)
        start=time.time()
        self.hdbscan_labels = self.hdbscan.fit_predict(self.data)
        end=time.time()
        self.hdbscan_time=end-start
    def exe_spect(self):
        self.spect = cluster.SpectralClustering(affinity='nearest_neighbors',assign_labels='discretize',
                                                n_clusters=self.spect_n_clusters,n_neighbors=self.spect_n_neighbors)
        start=time.time()
        self.spect_labels = self.spect.fit_predict(self.data)
        end=time.time()
        self.spect_time=end-start
    def exe_dp(self):
        self.dp =Density_Peaks_clustering(dens_type=self.dp_dens_type) 
        start=time.time()
        self.dp.get_decision_graph(self.data)
        self.dp.delta_cut=self.dp_delta_cut
        self.dp.dens_cut=self.dp_dens_cut
        self.dp.get_assignation(self.data)
        end=time.time()
        self.dp_time=end-start
