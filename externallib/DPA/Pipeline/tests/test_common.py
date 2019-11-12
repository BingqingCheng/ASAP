import pytest

from sklearn.utils.estimator_checks import check_estimator

from DPA import PointAdaptive_kNN
from DPA import twoNearestNeighbors
from DPA import DensityPeakAdvanced

@pytest.mark.parametrize(
    "Estimator", [DensityPeakAdvanced, PointAdaptive_kNN, twoNearestNeighbors] 
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
