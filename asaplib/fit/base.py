"""
Base class for regressions
"""

from sklearn.base import BaseEstimator, RegressorMixin
from .getscore import get_score


class RegressorBase(BaseEstimator, RegressorMixin):
    """
    Base class for regressions
    contains generic methods for computing errors
    """

    def __init__(self):
        """
        method specific
        """
        self._fitted = False

    def fit(self, X, y):
        """
        method specific
        """
        return self

    def predict(self, X, y):
        """
        method specific
        """
        pass

    def fit_predict(self, X, y, X_test):
        """
        Train the ridge regression model with the design matrix and trainLabel.
        Parameters
        ----------
        X : array-like, shape=[n_descriptors, n_samples]
        Input points.
        y : array-like, shape=[n_samples]
        Input points.
        X_test : array-like, shape=[n_descriptors, n_test_samples]
        Input points.
        """
        self.fit(X, y)
        return self.predict(X_test)

    def predict_error(self, X, y):
        """
        compute the score of the predictions compared with the true values
        Parameters
        ----------
        X : array-like, shape=[n_descriptors, n_samples]
        Input points.
        y : array-like, shape=[n_samples]
        Input points.
        """
        if not self._fitted:
            raise RuntimeError("The model has not been fitted yet, please fit it and then use predict.")

        # get the predictions for train set
        y_pred = self.predict(X)
        # compute the CV score
        y_error = get_score(y_pred, y)
        return y_pred, y_error

    def fit_predict_error(self, X, y, X_test, y_test):
        """
        Train the ridge regression model with the design matrix and trainLabel.
        and does prediction on test samples, and compute the error
        Parameters
        ----------
        X : array-like, shape=[n_descriptors, n_samples]
        Input points.
        y : array-like, shape=[n_samples]
        Input points.
        X_test : array-like, shape=[n_descriptors, n_test_samples]
        Input points.
        """
        self.fit(X, y)
        return self.predict_error(X_test, y_test)

    def get_train_test_error(self, X_train, y_train, X_test, y_test, verbose=True, return_pred=True):
        """
        train the model, and get the test error
        """
        fit_error = {}
        # compute the CV score for the train dataset
        y_pred, train_error = self.predict_error(X_train, y_train)
        if verbose: print("train score: ", train_error)
        fit_error['train_error'] = train_error

        # compute the CV score for the test dataset
        y_pred_test, test_error  = self.predict_error(X_test, y_test)
        if verbose: print("test score: ", test_error)
        fit_error['test_error'] = test_error

        if return_pred == True:
            return y_pred, y_pred_test, fit_error
        else:
            return fit_error

    def get_params(self, deep=True):
        pass

    def get_name(self):
        return type(self).__name__

    def set_params(self, deep=True):
        pass

    def pack(self):
        pass

    def unpack(self, state):
        pass

    def loads(self, state):
        pass


