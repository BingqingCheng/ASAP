"""
Tools for performing sparse kernel ridge regression of design matrices
using sparse KRR
NOTE: this class operate directly on design matrices.
for methods handingly kmatrix, see ./krr.py
"""

import numpy as np

from ..compressor import Sparsifier
from ..kernel import Descriptors_to_Kernels
from .base import RegressorBase

class SPARSE_KRR_Wrapper(RegressorBase):

    def __init__(self, kernel, krr_obj, sparse_mode="fps", n_sparse=None):
        """
        Parameters
        ----------
        kernel: dictionaries that specify which way to convert descriptors into kernel matrix
                     See ../kernel/kernel_transforms.py
        e.g.
        kernel = {
        "first_kernel": {"type": 'linear', "normalize": True}
        }
        e.g.
        {'k0':{"type": "cosine"}} 
        e.g.
        { 'k1': {"type": "polynomial", "d": power}}

        krr_obj: object for doing krr. Must have .fit() and .predict() methods
        sparsemode: str, default='fps', Sparsification method to use ([fps], [cur])'
        n_sparse: int, number of the representative samples, negative means no sparsification
         
        """
        self.sparse_mode = sparse_mode
        self.n_sparse = n_sparse
        if self.n_sparse is None or self.n_sparse > 0:
            self.sparsifier = Sparsifier(self.sparse_mode)
        else:
            self.sparsifier = None

        self.kernel = kernel
        # object for transform design matrix to kernel matrix
        self.k_transform = Descriptors_to_Kernels(kernel)

        # object for doing krr
        self.krr = krr_obj

        # we need to store the representative structures
        self.sbs = None
        # design matrix of the representative structures
        self.desc_sbs = None

        # state of the fit
        self._fitted = False

    def _sparsify(self, desc):
        # sparsification
        n_sample = len(desc)
        # set default value of n_sparse
        if self.n_sparse is None:
            self.n_sparse = max(10, n_sample // 20)
        # sparsification
        if self.n_sparse > 0 and self.n_sparse < n_sample:
            self.sbs = self.sparsifier.sparsify(desc, self.n_sparse)
        else:
            print("Not using any sparsification")
            self.sbs = range(n_sample)

        self.desc_sbs = desc[self.sbs]

    def fit(self, X, y):
        """
        Train the krr model with the design matrix and trainLabel.
        Parameters
        ----------
        X : array-like, shape=[n_descriptors, n_samples]
        Input points.
        y : array-like, shape=[n_samples]
        Input points.
        """
        print("a total of ", np.shape(X), "column")

        #if self._fitted:
        #    raise RuntimeError('SPARSE_KRR already fitted before, please reinitialise the object!')

        '''N train structures, M sparsified representative structures '''
        '''kMM: the kernel matrix of the representative structures with shape (M,M)'''
        '''kNM: the kernel matrix between the representative and the train structures with shape (N,M)'''
        self._sparsify(X)
        kMM = self.k_transform.compute(self.desc_sbs)
        kNM = self.k_transform.compute(X, self.desc_sbs)
        print(np.shape(kNM))
        self.krr.fit(kMM, y, kNM)

        self._fitted = True

    def predict(self, X):
        '''desc.shape is expected as [n_descriptors, n_samples]'''
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet, please fit it and then use predict.")
        kNM = self.k_transform.compute(X, self.desc_sbs)
        return self.krr.predict(kNM)

    def get_params(self, deep=True):
        pass

    def set_params(self, params, deep=True):
        pass

    def pack(self):
        pass

    def unpack(self, state):
        pass

    def loads(self, state):
        pass
