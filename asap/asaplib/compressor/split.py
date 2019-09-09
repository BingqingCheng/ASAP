"""
TODO: Module-level description
"""

from abc import ABCMeta, abstractmethod
import collections

import numpy as np
import scipy as sp
from sklearn.model_selection._split import KFold as _KFold
from sklearn.model_selection._split import ShuffleSplit as _ShuffleSplit
from sklearn.model_selection._split import (_BaseKFold,
        BaseCrossValidator,_validate_shuffle_split,BaseShuffleSplit)
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_random_state
from sklearn.externals.six import with_metaclass


def exponential_split(xmin, xmax, n=5):
    # obtain integeters that are equally spaced in log space
    X = np.zeros(n, dtype=int)
    [lmin, lmax] = [np.log(xmin), np.log(xmax)]
    dl = (lmax-lmin)/(n-1.)
    X[0] = xmin
    X[-1] = xmax
    for i in range(1, n-1):
        X[i] = int(np.exp(lmin+dl*i))
    return X


def kernel_random_split(X, y, r=0.05):

    if X.shape[0] != X.shape[1]:
        raise ValueError('K matrix is not a square')
    if len(X) != len(y):
        raise ValueError('Length of the vector of properties is not the same as number of samples')

    n_sample = len(X)
    all_list = np.arange(n_sample)
    randomchoice =  np.random.rand(n_sample)
    test_member_mask = (randomchoice < r)
    train_list = all_list[~test_member_mask]
    test_list = all_list[test_member_mask]

    X_train = X[:, train_list][train_list]
    y_train = y[train_list]

    X_test = X[:, train_list][test_list]
    y_test = y[test_list]
    return X_train, X_test, y_train, y_test, train_list, test_list

"""
adapt from Felix Musil ML_tools
"""


class KFold(_KFold):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(KFold, self).__init__(n_splits, shuffle, random_state)

    def get_params(self):
        params = dict(n_splits=self.n_splits,shuffle=self.shuffle,random_state=self.random_state)
        return params


class ShuffleSplit(_ShuffleSplit):
    def __init__(self, n_splits=10, test_size="default", train_size=None,random_state=None):
        super(ShuffleSplit, self).__init__(n_splits, test_size,train_size, random_state)
    def get_params(self):
        params = dict(n_splits=self.n_splits,test_size=self.test_size,
                    train_size=self.train_size,random_state=self.random_state)
        return params


class LCSplit(with_metaclass(ABCMeta)):
    def __init__(self, cv, n_repeats=[10], train_sizes=[10], test_size="default", random_state=None, **cvargs):
        if not isinstance(n_repeats, collections.Iterable) or not isinstance(train_sizes, collections.Iterable):
            raise ValueError("Number of repetitions or training set sizes must be an iterable.")

        if len(n_repeats) != len(train_sizes) :
            raise ValueError("Number of repetitions must be equal to length of training set sizes.")

        if any(key in cvargs for key in ('random_state', 'shuffle')):
            raise ValueError("cvargs must not contain random_state or shuffle.")

        self.cv = cv
        self.n_repeats = n_repeats
        self.train_sizes = train_sizes
        self.random_state = random_state
        self.cvargs = cvargs
        self.test_size = test_size
        self.n_splits = np.sum(n_repeats)
    
    def get_params(self):
        params = dict(cv=self.cv.get_params(), n_repeats=self.n_repeats,train_sizes=self.train_sizes,
                     test_size=self.test_size, random_state=self.random_state, cvargs=self.cvargs)
        return params

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        rng = check_random_state(self.random_state)
        
        for n_repeat, train_size in zip(self.n_repeats, self.train_sizes):
            cv = self.cv(random_state=rng, n_splits=n_repeat, test_size=self.test_size, train_size=train_size,
                         **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index
                
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        rng = check_random_state(self.random_state)
        n_splits = 0
        for n_repeat,train_size in zip(self.n_repeats, self.train_sizes):
            cv = self.cv(random_state=rng, n_splits=n_repeat, test_size=self.test_size, train_size=train_size,
                         **self.cvargs)
            n_splits += cv.get_n_splits(X, y, groups)
        return n_splits
