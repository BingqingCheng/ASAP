#!/usr/bin/python3
"""
Python script for performing ridge regression.
(with optional learning curve)
"""

import argparse
import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from asaplib.compressor import exponential_split, LCSplit, ShuffleSplit
from asaplib.data import ASAPXYZ, Design_Matrix
from asaplib.fit import RidgeRegression
from asaplib.io import str2bool
from asaplib.plot import plot_styles
from matplotlib import pyplot as plt

def main(fmat, fxyz, fy, prefix, scale, test_ratio, sigma, lc_points, lc_repeats):
    """

    Parameters
    ----------
    fmat: Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.
    fxyz: Location of xyz file for reading the properties.
    fy: Location of property list (1D-array of floats)
    prefix: filename prefix for learning curve figure
    scale: Scale the coordinates (True/False). Scaling highly recommanded.
    test_ratio: train/test ratio
    sigma: noise level in kernel ridge regression, default is 0.1% of the standard deviation of the data.
    lc_points : number of points on the learning curve
    lc_repeats : number of sub-sampling when compute the learning curve

    Returns
    -------

    Learning curve.

    """

    scale = bool(scale)

    # try to read the xyz file
    if fxyz != 'none':
        asapxyz = ASAPXYZ(fxyz)
        desc, _ = asapxyz.get_descriptors(fmat)
    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(fmat[0]):
        try:
            desc = np.genfromtxt(fmat[0], dtype=float)
            print("loaded the descriptor matrix from file: ", fmat)
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    if len(desc) == 0:
        raise ValueError('Please supply descriptor in a xyz file or a standlone descriptor matrix')
    print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))

    # read in the properties to be predicted
    y_all = []
    try:
        y_all = np.genfromtxt(fy, dtype=float)
    except:
        y_all = asapxyz.get_property(fy)

    dm = Design_Matrix(X=desc, y=y_all, whiten=True, test_ratio=test_ratio)

    # if sigma is not set...
    if sigma < 0:
        sigma = 0.001 * np.std(y_all)
    rr = RidgeRegression(sigma)

    # fit the model
    dm.compute_fit(rr, 'ridge_regression', store_results=True, plot=True)

    # learning curve
    if lc_points > 1:
        lc_scores = dm.compute_learning_curve(rr, 'ridge_regression', lc_points=lc_points, lc_repeats=lc_repeats, randomseed=42, verbose=False)
        # make plot
        lc_scores.plot_learning_curve()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', nargs='+', type=str, required=True,
                        help='Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-fy', type=str, default='none', help='Location of the list of properties (N floats)')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--scale', type=str2bool, nargs='?', const=True, default=True,
                        help='Scale the coordinates (True/False). Scaling highly recommanded.')
    parser.add_argument('--test', type=float, default=0.05, help='the test ratio')
    parser.add_argument('--sigma', type=float, default=-1,
                        help='the noise level of the signal. Also the regularizer that improves the stablity of matrix inversion.')
    parser.add_argument('--lcpoints', type=int, default=10,
                        help='the number of points on the learning curve, <= 1 means no learning curve')
    parser.add_argument('--lcrepeats', type=int, default=8,
                        help='the number of sub-samples to take when compute the learning curve')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.fy, args.prefix, args.scale, args.test, args.sigma, args.lcpoints, args.lcrepeats)
