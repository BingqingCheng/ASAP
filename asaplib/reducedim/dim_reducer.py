"""
Methods and functions to perform dimensionality reduction
"""
import numpy as np
import json
from copy import copy
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .sparse_kpca import SPARSE_KPCA
from .ml_pca import PCA
from ..io import NpEncoder


class Dimension_Reducers:
    def __init__(self, dreduce_spec_dict={}):
        """
        Object handing the dimensionality reduction of design matrices
        Parameters
        ----------
        dreduce_spec_dict: dictionaries that specify which dimensionality reduction algorithm to use 
        we can perform a list of D-reduction sequentially.
        e.g.
        dreduce_spec_dict = {
        "preprocessing": {"type": 'SCALE', 'parameter': None},
        "reduce2_umap": {"type": 'UMAP', 'parameter': {"n_components": 10, "n_neighbors": 10, "metric":'euclidean'}}
        }
        e.g.
        dreduce_spec_dict = {
        "reduce1_pca": {"type": 'PCA', 'parameter':{"n_components": 50, "scalecenter":True}},
        "reduce2_tsne": {"type": 'TSNE', 'parameter': {"n_components": 10, "perplexity":20}}
        }
        """
        self.dreduce_spec_dict = dreduce_spec_dict
        # list of Atomic_Descriptor objections
        self.engines = {}
        #self.acronym = "" 
        self._fitted = False
        self.bind()

    def add(self, dreduce_spec, tag):
        """
        adding the specifications of a new dimensionality reducer
        Parameters
        ----------
        dreduce_spec: a dictionary that specify which dimensionality reducer to use 
        """
        self.dreduce_spec_dict[tag] = dreduce_spec

    def pack(self):
        return json.dumps(self.dreduce_spec_dict, sort_keys=True, cls=NpEncoder)
    """
    def get_acronym(self):
        if self.acronym == "":
            for element in self.dreduce_spec_dict.keys():
                self.acronym += self.engines[element].get_acronym()
        return self.acronym
    """
    def bind(self):
        """
        binds the objects that actually performs dimension reduction
        these objects need to have .fit(desc)/.transform(desc)/.fit_transform(desc) methods to perform reduction on a design matrix desc
        """
        # clear up the objects
        self.engines = {}
        for element in self.dreduce_spec_dict.keys():
            self.engines[element] = self._call(self.dreduce_spec_dict[element])
            #self.dreduce_spec_dict[element]['acronym'] = self.engines[element].get_acronym()

    def _call(self, dreduce_spec):
        """
        call the specific descriptor objects
        """
        if "type" not in dreduce_spec.keys():
            raise ValueError("Did not specify the type of the dimensionality reducer.")
        elif dreduce_spec["type"] == "SCALE":
            print ("Using standard scaling of the data ...")
            return StandardScaler()
        elif dreduce_spec["type"] == "PCA":
            print ("Using PCA ...")
            return PCA(**dreduce_spec['parameter'])
        elif dreduce_spec["type"] == "SPARSE_KPCA":
            print ("Using kernel PCA (sparsified) ...")
            return SPARSE_KPCA(**dreduce_spec['parameter'])
        elif dreduce_spec["type"] == "UMAP":
            """https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.UMAP.fit_transform"""
            print ("Using UMAP ...")
            return umap.UMAP(**dreduce_spec['parameter'])
        elif dreduce_spec["type"] == "TSNE":
            print ("Using t-sne ...")
            """https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"""
            return TSNE(**dreduce_spec['parameter'])
        else:
            raise NotImplementedError 

    def fit(self, X):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        X : array-like, shape=[n_samples,n_dim_high]
            Input points.
        """
        for element in self.dreduce_spec_dict.keys():
            X = self.engines[element].fit(X)
        self._fitted = True

    def fit_transform(self, X):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        X : array-like, shape=[n_samples,n_dim_high]
            Input points.

        Returns
        -------
        x: array-like, shape=[n_samples,n_dim_low]
        """
        X = X.copy()
        for element in self.dreduce_spec_dict.keys():
            X = self.engines[element].fit_transform(X)
        self._fitted = True
        return X

    def transform(self, X):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        X : array-like, shape=[n_samples,n_dim_high]
            Input points.

        Returns
        -------
        x: array-like, shape=[n_samples,n_dim_low]
        """
        if self._fitted == False:
            raise ValueError("Haven't fit the dimensionality reducer. Use .fit or .fit_transform first")
        X = X.copy()
        for element in self.dreduce_spec_dict.keys():
            if self.dreduce_spec_dict[element]['type'] == 'TSNE':
                raise ValueError("TSNE does not allow out-of-sampling embedding")
            X = self.engines[element].transform(X)
        return X


