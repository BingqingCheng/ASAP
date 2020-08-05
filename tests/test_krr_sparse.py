import os

import numpy as np
from matplotlib import pyplot as plt

from asaplib.data import ASAPXYZ, Design_Matrix
from asaplib.fit import SPARSE_KRR_Wrapper, KRRSparse


def main():
    """

    Test if Ridge regression is working.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    prefix: string giving the filename prefix
    """
    fxyz = os.path.join(os.path.split(__file__)[0], 'small_molecules-SOAP.xyz')
    fmat = ['SOAP-n4-l3-c1.9-g0.23']
    fy = 'dft_formation_energy_per_atom_in_eV'
    prefix = "test-skrr"
    test_ratio = 0.05
    lc_points = 8
    lc_repeats = 8

    # try to read the xyz file
    asapxyz = ASAPXYZ(fxyz)
    desc, _ = asapxyz.get_descriptors(fmat, False)
    y_all = asapxyz.get_property(fy)
    # print(desc)

    dm = Design_Matrix(X=desc, y=y_all, whiten=True, test_ratio=test_ratio)

    # kernel, jitter, delta, sigma, sparse_mode="fps", n_sparse=None
    k_spec = {'k0': {"type": "linear"}}  # { 'k1': {"type": "polynomial", "d": power}}

    # if sigma is not set...
    sigma = 0.001 * np.std(y_all)
    krr = KRRSparse(0., None, sigma)
    skrr = SPARSE_KRR_Wrapper(k_spec, krr, sparse_mode="fps", n_sparse=-1)

    # fit the model
    dm.compute_fit(skrr, 'skrr', store_results=True, plot=True)

    # learning curve
    if lc_points > 1:
        dm.compute_learning_curve(skrr, 'ridge_regression', lc_points=lc_points, lc_repeats=lc_repeats, randomseed=42,
                                  verbose=False)

    dm.save_state(prefix)
    plt.show()


def test_gen(tmpdir):
    """Test the generation using pytest"""
    main()


if __name__ == '__main__':
    main()
