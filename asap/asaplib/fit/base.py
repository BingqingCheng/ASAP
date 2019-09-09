"""
TODO: Module-level description
"""


from sklearn.base import BaseEstimator, RegressorMixin


class RegressorBase(BaseEstimator, RegressorMixin):
    """
    TODO: class-level docstring
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        pass

    def predict_error(self, X, y=None):
        pass

    def get_params(self, deep=True):
        pass

    def get_name(self):
        return type(self).__name__


class TrainerBase(object):
    def __init__(self):
        pass

    def fit(self):
        pass
