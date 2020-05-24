"""
Tools for performing dimensionality reduction of design matrices
using sparse KPCA
"""

import numpy as np
from ..compressor import Sparsifier
from ..kernel import Descriptors_to_Kernels
from .ml_kpca import KernelPCA

class SPARSE_KPCA:
    def __init__(self, n_components=2, kernel={}, sparse_mode="fps", n_sparse=None):
        """
        Parameters
        ----------
        n_components : int, default=2
            number of dimensions to project to

        kernel: dictionaries that specify which way to convert descriptors into kernel matrix
                     See ../kernel/kernel_transforms.py`
        e.g.
        kernel = {
        "first_kernel": {"type": 'linear', "normalize": True}
        }
        e.g.
        {'k0':{"type": "cosine"}} 
        e.g.
        { 'k1': {"type": "polynomial", "d": power}}

        sparsemode: str, default='fps', 
                    possible method to use ([fps], [cur],[random],[sequential])'
        n_sparse: int, number of the representative samples, negative means no sparsification
         
        """

        # essential
        self.n_components = n_components
        self.sparse_mode = sparse_mode
        self.n_sparse = n_sparse
        self.kernel = kernel

        if self.n_sparse is None or self.n_sparse > 0:
            self.sparsifier = Sparsifier(self.sparse_mode)
        else:
            self.sparsifier = None

        # object for transform design matrix to kernel matrix
        self.k_transform = Descriptors_to_Kernels(kernel)
        # object for doing kpca
        self.kpca = KernelPCA(self.n_components)

        # we need to store the representative structures
        self.sbs = None
        # design matrix of the representative structures
        self.desc_sbs = None

        # state properties
        self._fitted = False

    def _sparsify(self, desc):
        # sparsification
        n_sample = len(desc)
        # set default value of n_sparse
        if self.n_sparse is None:
            self.n_sparse = max(10, n_sample // 20)
        # sparsification
        if self.n_sparse > 0:
            self.sbs = self.sparsifier.sparsify(desc, self.n_sparse)
        else:
            print("Not using any sparsification")
            self.sbs = range(n_sample)

        self.desc_sbs = desc[self.sbs]

    def fit(self, desc):
        """Fit KPCA on the precomputed descriptor matrix

        Parameters
        ----------
        desc: array-like, shape=[n_descriptors, n_samples]
        design matrix

        Returns
        -------
        """
        if self._fitted:
            raise RuntimeError('PCA already fitted before, please reinitialise the object!')

        self._sparsify(desc)

        kNN = self.k_transform.compute(self.desc_sbs)
        self.kpca.fit(kNN)

        self._fitted = True

    def transform(self, desc_test):
        """Transforms to the lower dimensions

        Parameters
        ----------
        desc_test : array_like, shape=[n_descriptors, n_samples]
            design matrix of the new samples

        Returns
        -------
        projected_vectors: numpy.array, shape=(L,n_components)
            projections of the design matrix on low dimension
        """
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet, please fit it and then use transform.")

        kNM = self.k_transform.compute(desc_test, self.desc_sbs)
        return self.kpca.transform(kNM)

    
    def fit_transform(self, desc):
        """Fit PCA on the design matrix & project to lower dimensions

        Parameters
        ----------
        desc : array_like, shape=[n_descriptors, n_samples]
            design matrix of the new samples

        Returns
        -------
        projected_vectors: numpy.array, shape=(L,n_components)
            projections of the design matrix on low dimension

        """
        self.fit(desc)
        return self.transform(desc)
