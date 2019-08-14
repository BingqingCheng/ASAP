import numpy as np
import scipy as sp
from sklearn.base import ClusterMixin

class ClusterBase(ClusterMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def get_cluster_labels(self,index=[]):
        pass
    def get_params(self,deep=True):
        pass
    def get_name(self):
        return type(self).__name__

class FitClusterBase(object):
    def __init__(self):
        pass
    def fit(self):
        pass
