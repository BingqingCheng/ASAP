"""
NOTE: These methods operate directly on the kernel matrix!!!
some functions are adapted from Felix Musil's ml_tools
"""

import numpy as np

from .base import RegressorBase
from .getscore import get_score

class KRR(RegressorBase):

    def __init__(self, jitter):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter  # noise level^2
        self.coninv = None  # inverse of the covariance matrix
        self._fitted = False

    def fit(self, kernel, y, kNM=[]):
        '''Train the krr model with trainKernel and trainLabel.'''

        reg = np.eye(kernel.shape[0]) * self.jitter
        self.coninv = np.linalg.inv(kernel + reg)
        self.alpha = np.linalg.solve(kernel + reg, y)
        self._fitted = True

    def predict(self, kernel):
        '''kernel.shape is expected as (nPred, nTrain)'''
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet, please fit it and then use predict.")
        return np.dot(kernel, self.alpha.flatten()).reshape((-1))

    def predict_uncertainty(self, k, delta):
        '''
        k.shape is expected as (nPred, nTrain), delta is the variance of y
        '''
        n_k = len(k)
        y_error = np.zeros(n_k)
        for i in range(n_k):
            y_error[i] = np.sqrt(delta * (1. - np.dot(k[i], np.dot(self.coninv, k[i]))))
        return y_error

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


class KRRSparse(RegressorBase):

    def __init__(self, jitter, delta, sigma):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter
        self.delta = None  # variance of the prior
        self.sigma = sigma  # noise
        self._fitted = False
        if self.jitter is None:
            self.jitter = 10e-20

    def fit(self, kMM, y, kNM):
        '''N train structures, M sparsified representative structures '''
        '''kMM: the kernel matrix of the representative structures with shape (M,M)'''
        '''kNM: the kernel matrix between the representative and the train structures with shape (N,M)'''

        # if (kMM.shape[0] != kMM.shape[1]):# or kMM.shape[0] != kNM.shape[1] or kNM.shape[0] != y.shape[0]):
        # raise ValueError('Shape of the kernel matrix is not consistent!')

        if self.delta is None:
            self.delta = np.std(y) / (np.trace(kMM) / len(kMM))

        if self.sigma is None:
            self.sigma = 0.001 * np.std(y)

        sparseK = kMM * self.delta * self.sigma ** 2 + np.dot(kNM.T, kNM) * self.delta ** 2
        sparseY = np.dot(kNM.T, y)
        reg = np.eye(kMM.shape[0]) * self.jitter

        self.alpha = np.linalg.solve(sparseK + reg, sparseY)
        self._fitted = True

    def predict(self, kNM):
        '''kNM: the kernel matrix between the representative and the new structures with shape (N,M)'''
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet, please fit it and then use predict.")
        return np.dot(self.delta ** 2 * kNM, self.alpha.flatten()).reshape((-1))

    def fit_predict(self, kMM, kNM, y, kNM_test):
        self.fit(kMM, kNM, y)
        return self.predict(kNM_test)

    def fit_predict_error(self, kMM, kNM, y, kNM_test, y_test):
        self.fit(kMM, kNM, y)
        return self.predict_error(kNM_test, y_test)

    def get_params(self, deep=True):
        return dict(jitter=self.jitter, delta=self.delta, sigma=self.sigma)

    def set_params(self, params, deep=True):
        self.jitter = params['jitter']
        self.sigma = params['sigma']
        self.alpha = None

    def pack(self):
        state = dict(weights=self.alpha, jitter=self.jitter, delta=self.delta, sigma=self.sigma)
        return state

    def unpack(self, state):
        self.alpha = state['weights']
        self.delta = state['delta']
        self.sigma = state['sigma']
        err_m = 'jitter are not consistent {} != {}'.format(self.jitter, state['jitter'])
        assert self.jitter == state['jitter'], err_m

    def loads(self, state):
        self.alpha = state['weights']
        self.jitter = state['jitter']
        self.delta = state['delta']
        self.sigma = state['sigma']


class KRRFastCV(RegressorBase):
    """ 
    taken from:
    An, S., Liu, W., & Venkatesh, S. (2007). 
    Fast cross-validation algorithms for least squares support vector machine and kernel ridge regression. 
    Pattern Recognition, 40(8), 2154-2162. https://doi.org/10.1016/j.patcog.2006.12.015
    """

    def __init__(self, jitter, delta, cv):
        self.jitter = jitter
        self.cv = cv
        self.delta = delta
        self._fitted = False

    def fit(self, kernel, y, kNM=[]):
        '''Fast cv scheme. Destroy kernel.'''
        np.multiply(self.delta ** 2, kernel, out=kernel)
        kernel[np.diag_indices_from(kernel)] += self.jitter
        kernel = np.linalg.inv(kernel)
        alpha = np.dot(kernel, y)
        Cii = []
        beta = np.zeros(alpha.shape)
        self.y_pred = np.zeros(y.shape)
        self.error = np.zeros(y.shape)
        for _, test in self.cv.split(kernel):
            Cii = kernel[np.ix_(test, test)]
            beta = np.linalg.solve(Cii, alpha[test])
            self.y_pred[test] = y[test] - beta
            self.error[test] = beta  # beta = y_true - y_pred

        del kernel
        self._fitted = True

    def predict(self, kernel=None):
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet, please fit it and then use predict.")
        '''kernel.shape is expected as (nPred, nTrain)'''
        return self.y_pred

    def get_params(self, deep=True):
        return dict(sigma=self.jitter, cv=self.cv)

    def set_params(self, params, deep=True):
        self.jitter = params['jitter']
        self.cv = params['cv']
        self.delta = params['delta']
        self.y_pred = None

    def pack(self):
        state = dict(y_pred=self.y_pred, cv=self.cv.pack(),
                     jitter=self.jitter, delta=self.delta)
        return state

    def unpack(self, state):
        self.y_pred = state['y_pred']
        self.cv.unpack(state['cv'])
        self.delta = state['delta']

        err_m = 'jitter are not consistent {} != {}'.format(self.jitter, state['jitter'])
        assert self.jitter == state['jitter'], err_m

    def loads(self, state):
        self.y_pred = state['y_pred']
        self.cv.loads(state['cv'])
        self.jitter = state['jitter']
        self.delta = state['delta']
