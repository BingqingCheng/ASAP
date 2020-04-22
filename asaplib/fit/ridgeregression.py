import numpy as np
import json

from .base import RegressorBase
from .getscore import get_score

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

    def get_error(self, desc, y):
        """
        compute the score of the predictions compared with the true values
        Parameters
        ----------
        desc : array-like, shape=[n_descriptors, n_samples]
        Input points.
        y : array-like, shape=[n_samples]
        Input points.
        """
        # get the predictions for train set
        y_pred = self.predict(desc)
        # compute the CV score
        y_error = get_score(y_pred, y)
        return y_error, y_pred

    def get_train_test_error(self, desc_train, y_train, desc_test, y_test, verbose=True, return_pred=True):
        fit_error = {}
        # compute the CV score for the train dataset
        train_error, y_pred = self.get_error(desc_train, y_train)
        if verbose: print("train score: ", train_error)
        fit_error['train_error'] = train_error

        # compute the CV score for the test dataset
        test_error, y_pred_test = self.get_error(desc_test, y_test)
        if verbose: print("test score: ", test_error)
        fit_error['test_error'] = test_error

        if return_pred == True:
            return fit_error, y_pred, y_pred_test
        else:
            return fit_error

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
