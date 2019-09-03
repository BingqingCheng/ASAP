
from sklearn.base import BaseEstimator, RegressorMixin,TransformerMixin


class RegressorBase(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def predict(self,X,y=None):
        pass

    def predict_error(self,X,y=None):
        pass

    def get_params(self,deep=True):
        pass

    def get_name(self):
        return type(self).__name__


class TrainerBase(object):
    def __init__(self):
        pass

    def fit(self):
        pass
