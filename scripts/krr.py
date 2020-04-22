#!/usr/bin/python3
"""
Python script for performing kernel ridge regression 
(with optional learning curve)
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from asaplib.compressor import exponential_split, LCSplit, ShuffleSplit
from asaplib.compressor import fps, kernel_random_split
from asaplib.data import ASAPXYZ
from asaplib.fit import KRRSparse, LC_SCOREBOARD
from asaplib.plot import plot_styles


def main(fmat, fxyz, fy, prefix, test_ratio, jitter, n_sparse, sigma, lc_points, lc_repeats):
    """

    Parameters
    ----------
    fmat: Location of kernel matrix file.
    fy: Location of property list (1D-array of floats)
    prefix: filename prefix for learning curve figure
    test_ratio: train/test ratio
    jitter: jitter level, default is 1e-10
    n_sparse: number of representative samples, default is 5% of the data
    sigma: noise level in kernel ridge regression, default is 0.1% of the standard deviation of the data.
    lc_points : number of points on the learning curve
    lc_repeats : number of sub-sampling when compute the learning curve

    Returns
    -------

    Fitting outcome & Learning curve.

    """

    # if it has been computed before we can simply load it
    try:
        K_all = np.genfromtxt(fmat, dtype=float)
    except OSError:
        raise Exception('fmat file could not be loaded. Please check the filename')
    print("loaded", fmat)

    # read in the properties to be predicted
    y_all = []
    try:
        y_all = np.genfromtxt(fy, dtype=float)
    except:
        try:
            # try to read the xyz file
            if fxyz != 'none':
                asapxyz = ASAPXYZ(fxyz)
                y_all = asapxyz.get_property(fy)
        except OSError:
            raise Exception('property vector file could not be loaded. Please check the filename')

    if len(y_all) != len(K_all):
        raise ValueError('Length of the vector of properties is not the same as number of samples')
    else:
        n_sample = len(K_all)

    # train test split
    if test_ratio > 0:
        K_train, K_test, y_train, y_test, _, _ = kernel_random_split(K_all, y_all, test_ratio)
    else:
        K_train = K_test = K_all
        y_train = y_test = y_all
    n_train = len(K_train)
    n_test = len(K_test)

    # set default value of n_sparse
    if n_sparse == 0:
        n_sparse = n_train // 20
    # sparsification
    if n_sparse >= n_train:
        print("the number of representative structure is too large, please select n < ", n_train)
    elif n_sparse > 0:
        ifps, dfps = fps(K_train, n_sparse, 0)
        K_MM = K_train[:, ifps][ifps]
        K_NM = K_train[:, ifps]
        K_TM = K_test[:, ifps]
    else:
        print("it's usually better to use some sparsification")
        K_MM = K_train
        K_NM = K_train
        K_TM = K_test

    # if sigma is not set...
    if sigma < 0:
        sigma = 0.001 * np.std(y_train)

    delta = np.std(y_train) / (np.trace(K_MM) / len(K_MM))
    krr = KRRSparse(jitter, delta, sigma)
    # fit the model
    krr.fit(K_MM, K_NM, y_train)

    fit_error = {}
    # get the predictions for train set
    y_pred = krr.predict(K_NM)
    # compute the CV score for the dataset
    y_pred, y_pred_test, fit_error = krr.get_train_test_error(K_NM, y_train, K_TM, y_test, verbose=True, return_pred=True)
    # dump to file
    import json
    with open('KRR_train_test_errors_4' + prefix + '.json', 'w') as fp:
        json.dump(fit_error, fp)

    # learning curve
    # decide train sizes
    if lc_points > 1 and n_sparse > 0:
        train_sizes = exponential_split(n_sparse, n_train - n_test, lc_points)
        print("Learning curves using train sizes: ", train_sizes)
        lc_stats = lc_repeats * np.ones(lc_points, dtype=int)
        lc = LCSplit(ShuffleSplit, n_repeats=lc_stats, train_sizes=train_sizes, test_size=n_test, random_state=10)

        lc_scores = LC_SCOREBOARD(train_sizes)
        for lctrain, _ in lc.split(y_train):
            Ntrain = len(lctrain)
            lc_K_NM = K_NM[lctrain, :]
            lc_y_train = y_train[lctrain]
            # here we always use the same test set
            # otherwise, one can do `lc_K_test = K_NM[lctest,:]; lc_y_test = y_train[lctest]`
            krr.fit(K_MM, lc_K_NM, lc_y_train)
            # here we always use the same test set
            _, lc_score_now = krr.fit_predict_error(K_MM, lc_K_NM, lc_y_train, K_TM, y_test)
            lc_scores.add_score(Ntrain, lc_score_now)

        sc_name = 'RMSE' #     MAE, RMSE, SUP, R2, CORR
        lc_results = lc_scores.fetch(sc_name)
        # output learning curve
        np.savetxt("KRR_learning_curve_4" + prefix + ".dat", lc_results)

    plot_styles.set_nice_font()

    if lc_points > 1 and n_sparse > 0:
        fig = plt.figure(figsize=(8 * 2.1, 8))
        ax = fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    ax.plot(y_train, y_pred, 'b.', label='train')
    ax.plot(y_test, y_pred_test, 'r.', label='test')
    ax.legend()
    ax.set_title('KRR for: ' + fy)
    ax.set_xlabel('actual y')
    ax.set_ylabel('predicted y')

    if lc_points > 1 and n_sparse > 0:
        ax2 = fig.add_subplot(122)
        ax2.errorbar(lc_results[:,0], lc_results[:,1], yerr=lc_results[:,2], linestyle='', uplims=True, lolims=True)
        ax2.set_title('Learning curve')
        ax2.set_xlabel('Number of training samples')
        ax2.set_ylabel('Test {}'.format(sc_name))
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    plt.show()
    fig.savefig('KRR_4_' + prefix + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, required=True,
                        help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-fy', type=str, default='none', help='Location of the list of properties (N floats)')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--test', type=float, default=0.05, help='the test ratio')
    parser.add_argument('--jitter', type=float, default=1e-10,
                        help='regularizer that improves the stablity of matrix inversion')
    parser.add_argument('--n', type=int, default=0,
                        help='number of the representative samples, set negative if using no sparsification')
    parser.add_argument('--sigma', type=float, default=1e-2, help='the noise level of the signal')
    parser.add_argument('--lcpoints', type=int, default=10,
                        help='the number of points on the learning curve, <= 1 means no learning curve')
    parser.add_argument('--lcrepeats', type=int, default=8,
                        help='the number of sub-samples to take when compute the learning curve')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.fy, args.prefix, args.test, args.jitter, args.n, args.sigma, args.lcpoints,
         args.lcrepeats)
