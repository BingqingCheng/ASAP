"""
Base classes for clustering algorithms.
"""

from sklearn.base import ClusterMixin


class ClusterBase(ClusterMixin):
    """
    Data structure to perform clustering and store data associated with the clustering output.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        return self

    def get_cluster_labels(self, index=[]):
        """

        Parameters
        ----------
        index

        Returns
        -------

        """
        pass

    def get_params(self, deep=True):
        """

        Parameters
        ----------
        deep

        Returns
        -------

        """
        pass

    def get_name(self):
        """

        Returns
        -------

        """
        return type(self).__name__


class FitClusterBase(object):
    def __init__(self):
        pass

    def fit(self, dmatrix, rho=None):
        """

        Parameters
        ----------
        dmatrix
        rho

        Returns
        -------

        """
        pass
