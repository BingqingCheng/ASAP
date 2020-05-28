"""
tools for doing PCA
e.g.
desc = np.genfromtxt(prefix+".desc")
proj = PCA(pca_d).fit_transform(desc)
"""

import numpy as np
import scipy.linalg as salg
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, n_components=2, scalecenter=True):
        """PCA with precomputed descriptor matrix

        Parameters
        ----------
        n_components : int, default=2
            number of dimensions to project to
        """

        # essential
        self.n_components = n_components
        # options for scale and center the descriptor matrix
        self.scalecenter = scalecenter
        self.scaler = None

        # eigenvectors for projections
        self.pvec = None

        # state properties
        self._fitted = False

    def scalecenter_matrix(self, desc):
        """Scaling and Centering of a design matrix, with additional centering info

        Parameters
        ----------
        desc : array-like, shape=[n_descriptors, n_samples]
            design matrix

        Returns
        -------
        """
        self.scaler = StandardScaler()
        print(self.scaler.fit(desc))
        return self.scaler.transform(desc)

    def centering(self, desc):
        # calculate the mean of each column
        M_desc = np.mean(desc.T, axis=1)
        # center columns by subtracting column means
        C_desc = desc - M_desc
        return C_desc

    def fit(self, desc):
        """Fit PCA on the precomputed descriptor matrix

        Parameters
        ----------
        desc: array-like, shape=[n_descriptors, n_samples]
        design matrix

        Returns
        -------
        """
        if self._fitted:
            raise RuntimeError('PCA already fitted before, please reinitialise the object!')

        print("Start PCA for a design matrix with shape ", np.shape(desc))

        # scale and center
        if self.scalecenter:
            C_desc = self.scalecenter_matrix(desc)
        else:
            # center columns by subtracting column means
            C_desc = self.centering(desc)

        # calculate covariance matrix of centered matrix
        COV = np.cov(C_desc.T)
        print("computing covariance matrix with shape:", np.shape(COV))

        print("  And now we build a projection ")
        eval, evec = salg.eigh(COV, eigvals=(len(COV) - self.n_components, len(COV) - 1))
        eval = np.flipud(eval)
        evec = np.fliplr(evec)

        self.pvec = evec.copy()
        for i in range(self.n_components):
            self.pvec[:, i] *= 1. / np.sqrt(eval[i])
        print("Done, super quick. ")

        # the model is fitted then
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

        if self.scalecenter:
            desc_test = self.scaler.transform(desc_test)

        return np.dot(desc_test, self.pvec)

    def fit_transform(self, desc):
        """Fit PCA on the design matrix & project to lower dimensions

        Parameters
        ----------
        desc_test : array_like, shape=[n_descriptors, n_samples]
            design matrix of the new samples

        Returns
        -------
        projected_vectors: numpy.array, shape=(L,n_components)
            projections of the design matrix on low dimension

        """
        self.fit(desc)
        return self.transform(desc)
