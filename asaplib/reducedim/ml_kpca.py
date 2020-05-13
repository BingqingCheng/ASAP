"""
Tools for doing kernel PCA on environmental similarity
e.g.
kNN = np.genfromtxt(prefix+".k",skip_header=1)
proj = KernelPCA(kpca_d).fit_transform(kNN)

KernelPCA with precomputed kernel for now only!!!!!!!!!!!!
"""

import numpy as np
import scipy.linalg as salg


class KernelPCA:
    def __init__(self, n_components=2):
        """KernelPCA with precomputed kernel for now only

        Parameters
        ----------
        n_components : int, default=2
            number of dimensions to project to
        """

        # essential
        self.n_components = n_components

        # kernel and projection related properties
        self.vectors = None
        self.colmean = None
        self.mean = None
        self.center_kmat = None
        self._lambdas = None
        self._alphas = None
        self._m = None

        # state properties
        self._fitted = False

    @staticmethod
    def center_square(kernel):
        """Centering of a kernel matrix, with additional centering info

        Parameters
        ----------
        kernel : array-like, (shape=MxM)
            Kernel matrix

        Returns
        -------
        colmean : ndarray, (shape=M)
            mean of columns from kernel matrix
        mean : float
            mean of the entire kernel matrix
        centered_kernel : ndarray, (shape=MxM)
            kernel centered in feature space
        """
        colmean = np.mean(kernel, axis=1)
        mean = np.mean(colmean)
        centered_kernel = kernel - colmean - np.atleast_2d(colmean).T + mean
        return colmean, mean, centered_kernel

    def _center_test_kmat(self, ktest):
        """Centers a kernel matrix for projection

        Parameters
        ----------
        ktest : array_like, shape=(L,M)
            kernel matricx for the points to project to the lower dimensions
            Overwritten!!!

        Returns
        -------
        ktest :
            centered kernel matrix

        """
        test_colmean = np.atleast_2d(np.mean(ktest, axis=1)).T
        ktest += self.mean - self.colmean - test_colmean
        return ktest

    def fit(self, kmat):
        """Fit kernel PCA on the precomputed kernel matrix

        Notes
        -----
        - Keeps the kernel matrix intact
        - can be done only once on a given object, need to reinitialise if trying to run again

        Parameters
        ----------
        kmat : numpy.ndarray, shape=(M,M)
            kerenl matrix between the observations
            needs to be square and real, symmetric

        Returns
        -------
        """
        if self._fitted:
            raise RuntimeError('Kernel already fitted before, please reinitialise the object!')

        # check the shape and symmetry of the kernel matrix
        self._check_kmat(kmat, square=True)

        # save kmat and center
        self._m = len(kmat)
        self.colmean, self.mean, self.center_kmat = self.center_square(kmat)

        # calculation, ordering in descending order and scale by sqrt(lambda)
        self._lambdas, self._alphas = salg.eigh(self.center_kmat, eigvals=(self._m - self.n_components, self._m - 1))
        self._lambdas = np.flipud(self._lambdas)
        self._alphas = np.fliplr(self._alphas) / np.sqrt(self._lambdas)

        # the model is fitted then
        self._fitted = True

    def fit_transform(self, kmat, copy=True):
        """Fit kernel PCA on the precomputed kernel matrix & project to lower dimensions

        Notes
        -----
        - Keeps the kernel matrix intact
        - can be done only once on a given object, need to reinitialise if trying to run again

        Parameters
        ----------
        kmat : numpy.ndarray, shape=(M,M)
            kerenl matrix between the observations
            needs to be square and real, symmetric
        copy : bool, optional, default=True
            copy the kernel matrix or overwrite it, passed to self.transform()
            nb. the kernel matrix will be left centered if this is False

        Returns
        -------

        """
        self.fit(kmat)
        return self.transform(self.center_kmat, iscentered=True, copy=copy)

    def transform(self, ktest, iscentered=False, copy=True):
        """Transforms to the lower dimensions

        Parameters
        ----------
        ktest : array_like, shape=(L,M)
            kernel matrix of the test vectors with the training vectors
        iscentered : bool, optional, default=False
            if the kernel is centered already, mainly used for the fit_transform function
        copy : bool, optional, default=True
            copy the kernel matrix or overwrite it
            nb. the kernel matrix will be left centered if this is False

        Returns
        -------
        projected_vectors: numpy.array, shape=(L,n_components)
            projections of the given kernel wrt. the training kernel
        """
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet, please fit it and then use transform.")

        self._check_kmat(ktest, square=False)

        if iscentered:
            centered_ktest = ktest
        else:
            centered_ktest = self._center_test_kmat((ktest.copy() if copy else ktest))

        return np.dot(centered_ktest, self._alphas)

    def _check_kmat(self, kmat, square=True):
        """Check the kernel matrix

        checks if the matrix is real valued and its shape: square symmetric if square=True, LxM(self._m) otherwise

        Parameters
        ----------
        kmat : numpy.array
            kernel matrix to be checked
        square : bool, default=True
            if the matrix needs to be square and symmetric

        Returns
        -------

        """

        assert np.ndim(kmat) == 2
        assert np.isreal(kmat).all()

        if square:
            # shape (M,M)
            assert np.shape(kmat)[0] == np.shape(kmat)[1]
            # symmetric
            assert (kmat - kmat.T).max() < 1e-6
        else:
            # shape: (anything,M)
            assert np.shape(kmat)[1] == self._m

    def fit_vectors(self, vecs):
        """Fit Kernel PCA from vectors in the large dimension

        for future, not implemented yet
        """
        raise NotImplementedError

    def transform_vectors(self, vecs):
        """Fit Kernel PCA from vectors in the large dimension & project to lower dimension

        for future, not implemented yet
        """
        raise NotImplementedError
