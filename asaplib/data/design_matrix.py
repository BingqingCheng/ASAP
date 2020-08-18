"""
Class for storing and handling design matrices
"""

import numpy as np
import json
from yaml import dump as ydump

from ..io import randomString,  NpEncoder
from ..compressor import random_split,exponential_split, LCSplit, ShuffleSplit
from ..compressor import Sparsifier
from ..fit import LC_SCOREBOARD

class Design_Matrix:
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
    def __init__(self, X=[], y=[], whiten=True, test_ratio=0, random_seed=42, z=[], tags=[]):
        # sanity checks
        if len(y) > 0 and len(X) != len(y):
            raise ValueError('Length of the labels y is not the same as the design matrix X')
        
        if whiten:
            X = self._whiten(X)

        self.n_sample = len(X)
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # We do a train/test split already here, to ensure the subsequent analysis is prestine
        self.train_list, self.test_list = random_split(self.n_sample, test_ratio, random_seed)
        self.X_train = X[self.train_list]
        self.X_test = X[self.test_list]
        self.n_train = len(self.X_train)
        self.n_test = len(self.X_test)

        if len(y) > 0:
            self.y_train = y[self.train_list]
            self.y_test = y[self.test_list]
        if len(z) > 0:
            self.z_train = z[self.train_list]
            self.z_test = z[self.test_list]
        if len(tags) > 0:
            self.tags_train = tags[self.train_list]
            self.tags_test = tags[self.test_list]

        # this stores the index of representative data
        self.sbs = range(self.n_train)
        self.n_sparse = len(self.X_train) # no sparsification for now

        # these store the results of the fit using external learners
        self.fit_error_by_learner = {} # fitting errors
        self.lc_by_learner = {} # learning curves

    def _whiten(self, X, scale = True, addbias = True):
        """ scale & center """
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

    def save_state(self, filename, mode='yaml'):
        """output json or yaml file"""
        if mode == 'yaml':
            with open(filename+'-fit-errors.yaml', 'w') as yd:
                ydump(self.fit_error_by_learner, yd, sort_keys=True)
            with open(filename+'-lc.yaml', 'w') as yd:
                ydump(self.lc_by_learner, yd, sort_keys=True)
        else:
            with open(filename+'-fit-errors.json', 'w') as jd:
                json.dump(self.fit_error_by_learner, jd, sort_keys=True, cls=NpEncoder)
            with open(filename+'-lc.json', 'w') as jd:
                json.dump(self.lc_by_learner, jd, sort_keys=True, cls=NpEncoder)

    def sparsify(self, n_sparse=None, sparse_mode='fps'):
        """
        select representative data points using the design matrix

        Parameters
        ----------
        n_sparse: int
                  number of representative points
                  n_sparse == None means 5% of the data
                  n_sparse < 0 means no sparsification
        sparse_mode: str
                  Methods to use for sparsification [cur], [fps], [random]
        """
        
        # set default value of n_sparse
        if n_sparse is None:
            n_sparse = max(10, self.n_train // 20)

        # sparsification
        if n_sparse > 0:
            sparsifier = Sparsifier(sparse_mode)
            self.sbs = sparsifier.sparsify(desc, n_sparse)

    def get_sparsified_matrix(self):
        if len(self.y_train) > 0:
            return self.X_train[self.sbs], self.y_train[self.sbs]
        else:
            return self.X_train[self.sbs], []

    def compute_fit(self, learner, tag=None, store_results=True, plot=True):
        """
        Fit the design matrix X and the values y using a learner

        Parameters
        ----------
        learner: a learner object
                 e.g. RidgeRegression
                 needs to have .fit(), .predict(), .get_train_test_error(), .fit_predict_error() methods
        tag: str
              The name of this learner
        """
        # sanity checks
        if len(self.y_train) < 1 or len(self.X_train) != len(self.y_train):
            raise ValueError('Length of the labels y is not the same as the design matrix X')

        if tag is None: tag = randomString(6)

        # fit the model
        learner.fit(self.X_train[self.sbs], self.y_train[self.sbs])

        if store_results:
            y_pred, y_pred_test, fit_error = learner.get_train_test_error(self.X_train[self.sbs], self.y_train[self.sbs], self.X_test, self.y_test, verbose=True, return_pred=True)
            self.fit_error_by_learner[tag] = {"error": fit_error}

        if plot:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(self.y_train[self.sbs], y_pred, 'b.', label='train')
            ax.plot(self.y_test, y_pred_test, 'r.', label='test')
            ax.legend()
            ax.set_title('Fits using: '+str(tag))
            ax.set_xlabel('actual y')
            ax.set_ylabel('predicted y')
            return fig, ax

    def compute_learning_curve(self, learner, tag=None, lc_points=8, lc_repeats=8, randomseed=42, verbose=True):
        """
        Fit the learning curve using a learner

        Parameters
        ----------
        lc_points: int
                   the number of points on the learning curve
        lc_repeats: int
                the number of sub-samples to take when compute the learning curve
        learner: a learner object
                 e.g. RidgeRegression
                 needs to have .fit(), .predict(), .get_train_test_error(), .fit_predict_error() methods
        tag: str
                The name of this learner
        """
        if tag is None: tag = randomString(6)

        if lc_points < 1:
            print("The number of points on the learning curve < 1. Skip computing learning curve")
            return
        if lc_repeats < 1:
            print("The number sub-samples to take when compute the learning curve < 1. Skip computing learning curve")
            return

        # set lower and upper bound of the LC
        train_sizes = exponential_split(max(20,self.n_train//1000), len(self.sbs), lc_points)
        print("Learning curves using train sizes: ", train_sizes)
        lc_stats = lc_repeats * np.ones(lc_points-1, dtype=int)
        lc = LCSplit(ShuffleSplit, n_repeats=lc_stats, train_sizes=train_sizes[:-1], test_size=None, random_state=randomseed)

        lc_scores = LC_SCOREBOARD(train_sizes)
        for lctrain, _ in lc.split(self.X_train[self.sbs]):
            Ntrain = len(lctrain)
            lc_X_train = self.X_train[self.sbs][lctrain, :]
            lc_y_train = self.y_train[self.sbs][lctrain]
            # here we always use the same test set
            _, lc_score_now = learner.fit_predict_error(lc_X_train, lc_y_train, self.X_test, self.y_test)
            lc_scores.add_score(Ntrain, lc_score_now)
        # for the whole data set
        _, lc_score_now = learner.fit_predict_error(self.X_train[self.sbs], self.y_train[self.sbs], self.X_test, self.y_test)
        lc_scores.add_score(self.n_train, lc_score_now)

        self.lc_by_learner.update({tag:lc_scores.fetch_all()})
        if verbose: print("LC results: ", {tag:lc_scores.fetch_all()})
        return lc_scores


