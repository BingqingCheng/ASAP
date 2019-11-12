import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy.testing as npt

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from Pipeline.PAk import PointAdaptive_kNN


@pytest.fixture
def data_Fig1():
    # Read dataset used for Figure 1 in the paper.
    data_F1 = pd.read_csv("./benchmarks/Fig1.dat", sep=" ", header=None)
    return data_F1
    #return load_iris(return_X_y=True)

@pytest.fixture
def output_Fig1_test1():
    # Read benchmark output of the PAk algorithm: right after the search of the
    # optimal k_hat for each point.
    out_F1 = pd.read_csv("./benchmarks/output_Fig1_test1.csv", header=None)
    out_F1.columns = ["rho","rho_err","k_hat","dc"]
    return out_F1

@pytest.fixture
def output_Fig1_test2():
    # Read benchmark output of the PAk algorithm: right after the search of the
    # shift in the densities values.
    out_F1 = pd.read_csv("./benchmarks/output_Fig1_test2.csv", header=None)
    out_F1.columns = ["rho","rho_err","k_hat","dc"]
    return out_F1

@pytest.fixture
def output_Fig1_test3():
    # Read benchmark output of the PAk algorithm: right after the search of the
    # correction for bias and the shift in the densities values.
    out_F1 = pd.read_csv("./benchmarks/output_Fig1_test3.csv", header=None)
    out_F1.columns = ["rho","rho_err","k_hat","dc"]
    return out_F1

def is_almost_equal(x,y,mismatch, decimal):
    d = 0
    for i in range(len(x)):
        if abs(x[i]-y[i]) > 1.5 * 10**(-decimal):
            d += 1
    print(d/len(x)*100)
    if d/len(x)*100>mismatch:
        npt.assert_almost_equal(x, y, decimal=decimal)
    else:
        assert True

def test_PointAdaptive_kNN(data_Fig1, output_Fig1_test3):
    est = PointAdaptive_kNN(n_jobs=-1, dim_algo="twoNN")
    assert est.dim == None
    assert est.k_max == 1000
    assert est.D_thr == 23.92812698
    assert est.metric == "euclidean"
    assert est.dim_algo == "twoNN"    

    est.fit(data_Fig1)
    assert hasattr(est, 'is_fitted_')

    assert est.k_max == max(est.k_hat_)+1 # k_max include the point i
    assert len(data_Fig1) == len(est.densities_)

    df_bm = output_Fig1_test3
    is_almost_equal(est.densities_, df_bm['rho'].values, 0.07, 2)
    is_almost_equal(est.err_densities_, df_bm['rho_err'].values, 0.05, 3)
    is_almost_equal(est.dc_, df_bm['dc'].values, 0.05, 3)
    is_almost_equal(est.k_hat_, df_bm['k_hat'].values, 0.05, 0)

    df_results = pd.DataFrame(index=range(0,len(est.densities_)), columns=["rho","rho_err","k_hat","dc"])
    df_results["rho"] = est.densities_
    df_results["rho_err"] = est.err_densities_
    df_results["k_hat"] = est.k_hat_
    df_results["dc"] = est.dc_

    print(df_results.head())
    print(df_bm.head())
    #assert_frame_equal(df_results,df_bm)


