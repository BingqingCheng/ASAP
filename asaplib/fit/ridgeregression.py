import numpy as np

from .base import RegressorBase


class RidgeRegression(RegressorBase):
    _pairwise = True

    def __init__(self, jitter):
        # Weights of the rr model
        self.alpha = None
        self.jitter = jitter  # noise level^2
        self.coninv = None  # inverse of the covariance matrix

    def fit(self, desc, y):
        """
        Train the ridge regression model with the design matrix and trainLabel.
        Parameters
        ----------
        desc : array-like, shape=[n_descriptors, n_samples]
        Input points.
        y : array-like, shape=[n_samples]
        Input points.
        """

        print("a total of ", np.shape(desc), "column")

        # calculate covariance matrix
        COV = np.dot(desc.T, desc)
        print("computing covariance matrix with shape:", np.shape(COV))

        reg = np.eye(COV.shape[0]) * self.jitter
        # self.coninv = np.linalg.inv(COV+reg)
        # self.alpha = np.dot(self.coninv, np.dot(desc.T,y))
        self.alpha = np.linalg.solve(COV + reg, np.dot(desc.T, y))

    def predict(self, desc):
        '''kernel.shape is expected as (nPred, nTrain)'''
        return np.dot(desc, self.alpha.flatten()).reshape((-1))

    def get_params(self, deep=True):
        return dict(sigma=self.jitter)

    def set_params(self, params, deep=True):
        self.jitter = params['jitter']
        self.alpha = None

    def pack(self):
        state = dict(weights=self.alpha, jitter=self.jitter)
        return state

    def unpack(self, state):
        self.alpha = state['weights']
        err_m = 'jitter are not consistent {} != {}'.format(self.jitter, state['jitter'])
        assert self.jitter == state['jitter'], err_m

    def loads(self, state):
        self.alpha = state['weights']
        self.jitter = state['jitter']
