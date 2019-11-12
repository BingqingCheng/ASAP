import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy.testing as npt

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from Pipeline.twoNN import twoNearestNeighbors


@pytest.fixture
def data_Fig1():
    # Read dataset used for Figure 1 in the paper.
    data_F1 = pd.read_csv("./benchmarks/Fig1.dat", sep=" ", header=None)
    return data_F1
    #return load_iris(return_X_y=True)

@pytest.fixture
def output_xy_test1():
    # Read benchmark output of the TWO-NN algorithm: x and y used for the fit over the whole data set
    out_F1 = pd.read_csv("./benchmarks/output_xy_test1.csv", header=None)
    out_F1.columns = ["x","y"]
    return out_F1

def test_twoNN(data_Fig1, output_xy_test1):
    est = twoNearestNeighbors(frac=0.8, n_jobs=-1)
    assert est.blockAn == True
    assert est.block_ratio == 20
    assert est.metric == "euclidean"
    #assert est.frac == 1    

    est.fit(data_Fig1)
    assert hasattr(est, 'is_fitted_')

    df_bm = output_xy_test1
    #npt.assert_almost_equal(est.x_, df_bm['x'], decimal=5)
    #npt.assert_almost_equal(est.y_, df_bm['y'], decimal=5)
    
    assert int(round(est.dim_)) == 2

