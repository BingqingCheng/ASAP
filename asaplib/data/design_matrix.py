"""
Class for storing and handling design matrices
"""

import numpy as np
import json

from ..io import randomString,  NpEncoder
from ..compressor import random_split,exponential_split, LCSplit, ShuffleSplit
from ..compressor import fps, CUR_deterministic

class Design_Matrix:
    def __init__(self, X=[], y=[], whiten=True, testratio=0, randomseed=42, z=[], tags=[]):
        """extended design matrix class

        Parameters
        ----------
        X : array-like, shape=[n_samples,n_desc]
        Input points.
        y : array-like, shape=[n_samples]
        label for every point
        testratio: float, ratio of the test fraction
        z : array-like, shape=[n_samples]
        additional label for every point
        tags: array-like, strings, shape=[n_samples]
        additional tags for each data point
        """
        # sanity checks
        if len(y) > 0 and len(X) != len(y):
            raise ValueError('Length of the labels y is not the same as the design matrix X')
          
        if whiten:
            X = self._whiten(X)

        # We do a train/test split already here, to ensure the subsequent analysis is prestine
        self.train_list, self.test_list = random_split(n_sample, testratio, randomseed)
        self.X_train = X[train_list]
        self.X_test = X[test_list]
        self.nsamples = len(X)
        self.n_train = len(self.X_train)
        self.n_test = len(self.X_test)

        if len(y) > 0:
            self.y_train = y[train_list]
            self.y_test = y[test_list]
        if len(z) > 0:
            self.z_train = z[train_list]
            self.z_test = z[test_list]
        if len(tags) > 0:
            self.tags_train = tags[train_list]
            self.tags_test = tags[test_list]

        # this stores the index of representative data
        self.sbs = range(self.n_train)
        self.n_sparse = len(self.X_train) # no sparsification for now

        # these store the results of the fit using external learners
        self.fit_error_by_learner = {} # fitting errors
        self.lc_by_learner = {} # learning curves

    def _whiten(self, X, scale = True, addbias = True):
        # scale & center
        if scale:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)  # normalizing the features
        # add bias
        if addbias:
            X_bias = np.ones((np.shape(X)[0], np.shape(X)[1] + 1))
            X_bias[:, 1:] = X
            return X_bias
        else:
            return X

    def sparsify(self, n_sparse=0, sparse_mode='fps'):
        """
        select representative data points using the design matrix

        Parameters
        ----------
        n_sparse: number of representative points
                  n_sparse < 0 means 5% of the data
                  n_sparse == 0 means no sparsification
        sparse_mode: str. Methods to use for sparsification [cur], [fps], [random]
        """
        # sanity check
        if n_sparse >= self.n_train:
            print("the number of representative structure is too large, please select n < ", self.n_train)
        # set default value of n_sparse
        if n_sparse < 0:
            n_sparse = max(10, self.n_train // 20)
    
        # no sparsification
        if n_sparse == 0:
            self.sbs = range(self.n_train)
            self.n_sparse = len(self.X_train)
        else:
            self.n_sparse = n_sparse
            # sparsification
            if sparse_mode == 'fps' or sparse_mode == 'FPS':
                self.sbs, _ = fps(self.X_train, self.n_sparse, 0)
            elif sparse_mode == 'cur' or sparse_mode == 'CUR':
                cov = np.dot(np.asmatrix(self.X_train), np.asmatrix(self.X_train).T)
                self.sbs, _ = CUR_deterministic(cov, self.n_sparse)
            elif sparse_mode == 'random' or sparse_mode == 'RANDOM' or sparse_mode == 'Random':
                _, self.sbs = random_split(len(self.X_train), self.n_sparse/len(self.X_train))
            else:
                raise NotImplementedError("Do not recognize the selected sparsification mode. Use ([cur], [fps], [random]).")

    def get_sparsified_matrix(self):
        if len(self.y_train) > 0:
            return self.X_train[self.sbs], self.y_train[self.sbs]
        else:
            return self.X_train[self.sbs], []

    def compute_fit(self, learner, tag=None, store_results=True):
        """
        Fit the design matrix X and the values y using a learner

        Parameters
        ----------
        learner: object. a learner object, e.g. RidgeRegression
                 needs to have .fit(), .predict(), .get_train_test_error() methods
        tag: str. The name of this learner
        """
        if tag is None: tag = randomString(6)

        # fit the model
        learner.fit(X_train, y_train)

        if store_results:
            y_pred, y_pred_test, fit_error = learner.get_train_test_error(X_train, y_train, X_test, y_test, verbose=True, return_pred=True)
            self.fit_error_by_learner.update(tag:{"y_pred": y_pred, "y_pred_test": y_pred_test, "error": fit_error})

    def compute_learning_curve(self, learner, tag=None, lc_points=8, lc_repeats=8, store_results=True):
        """
        Fit the learning curve using a learner

        Parameters
        ----------
        lc_points: int, the number of points on the learning curve
        lc_repeats: the number of sub-samples to take when compute the learning curve
        learner: object. a learner object, e.g. RidgeRegression
                 needs to have .fit(), .predict(), .get_train_test_error() methods
        tag: str. The name of this learner
        """
        if tag is None: tag = randomString(6)

        if lc_points < 1:
            print("The number of points on the learning curve < 1. Skip computing learning curve")
            return
        if lc_repeats < 1:
            print("The number sub-samples to take when compute the learning curve < 1. Skip computing learning curve")
            return

        train_sizes = exponential_split(20, n_train - n_test, lc_points)
        print("Learning curves using train sizes: ", train_sizes)
        lc_stats = lc_repeats * np.ones(lc_points, dtype=int)
        lc = LCSplit(ShuffleSplit, n_repeats=lc_stats, train_sizes=train_sizes, test_size=n_test, random_state=10)

        lc_scores = LC_SCOREBOARD(train_sizes)
        for lctrain, _ in lc.split(y_train):
            Ntrain = len(lctrain)
            lc_X_train = X_train[lctrain, :]
            lc_y_train = y_train[lctrain]
            # here we always use the same test set
            _, lc_score_now = rr.fit_predict_error(lc_X_train, lc_y_train, X_test, y_test)
            lc_scores.add_score(Ntrain, lc_score_now)


