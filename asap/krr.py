#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from asaplib.compressor import fps, kernel_random_split
from asaplib.compressor import exponential_split, LCSplit, ShuffleSplit
from asaplib.fit import KRRSparse
from asaplib.fit import get_score
from asaplib.plot import plot_styles


def main(fmat, fy, prefix, test_ratio, jitter, n_sparse, sigma):

    """

    Parameters
    ----------
    fmat: Location of kernel matrix file.
    fy: Location of property list (1D-array of floats)
    prefix: filename prefix for learning curve figure
    test_ratio: train/test ratio
    jitter: jitter level, default is 1e-10
    n_sparse: number of representative samples
    sigma: noise level in kernel ridge regression

    Returns
    -------

    Learning curve.

    """

    # if it has been computed before we can simply load it
    try:
        K_all = np.genfromtxt(fmat, dtype=float)
    except OSError:
        raise Exception('fmat file could not be loaded. Please check the filename')
    print("loaded", fmat)
    try:
        y_all = np.genfromtxt(fy, dtype=float)
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
        
    delta = np.std(y_train)/(np.trace(K_MM)/len(K_MM))
    krr = KRRSparse(jitter, delta, sigma)
    # fit the model
    krr.fit(K_MM, K_NM, y_train)

    # get the predictions for train set
    y_pred = krr.predict(K_NM)
    # compute the CV score for the dataset
    print("train score: ", get_score(y_pred, y_train))
    # get the predictions for test set
    y_pred_test = krr.predict(K_TM)
    # compute the CV score for the dataset
    print("test score: ", get_score(y_pred_test, y_test))

    plot_styles.set_nice_font()
    fig = plt.figure(figsize=(8*2.1, 8))
    ax = fig.add_subplot(121)
    ax.plot(y_train, y_pred, 'b.', label='train')
    ax.plot(y_test, y_pred_test, 'r.', label='test')
    ax.legend()
    ax.set_title('KRR for: '+fy)
    ax.set_xlabel('actual y')
    ax.set_ylabel('predicted y')

    # learning curve
    # decide train sizes
    lc_points = 10
    train_sizes = exponential_split(n_sparse, n_train-n_test, lc_points)
    print("Learning curves using train sizes: ", train_sizes)
    lc_stats = 12*np.ones(lc_points, dtype=int)
    lc = LCSplit(ShuffleSplit, n_repeats=lc_stats, train_sizes=train_sizes, test_size=n_test, random_state=10)

    scores = {size: [] for size in train_sizes}
    for lctrain, lctest in lc.split(y_train):
        Ntrain = len(lctrain)
        lc_K_NM = K_NM[lctrain, :]
        lc_y_train = y_train[lctrain]
        #lc_K_test = K_NM[lctest,:]
        lc_K_test = K_TM
        #lc_y_test = y_train[lctest]
        lc_y_test = y_test
        krr.fit(K_MM, lc_K_NM, lc_y_train)
        lc_y_pred = krr.predict(lc_K_test)
        scores[Ntrain].append(get_score(lc_y_pred,lc_y_test))

    sc_name = 'RMSE'
    Ntrains = []
    avg_scores = []
    avg_scores_error = []
    for Ntrain, score in scores.items():
        avg = 0.
        var = 0.
        for sc in score:
            avg += sc[sc_name]
            var += sc[sc_name]**2.
        avg /= len(score)
        var /= len(score); var -= avg**2.
        avg_scores.append(avg)
        avg_scores_error.append(np.sqrt(var))
        Ntrains.append(Ntrain)

    ax2 = fig.add_subplot(122)
    ax2.errorbar(Ntrains, avg_scores, yerr=avg_scores_error)
    ax2.set_title('Learning curve')
    ax2.set_xlabel('Number of training samples')
    ax2.set_ylabel('Test {}'.format(sc_name))
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()
    fig.savefig('KRR_4_'+prefix+'.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, required=True, help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-y', type=str, default='none', help='Location of the list of properties (N floats)')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--test', type=float, default=0.0, help='the test ratio')
    parser.add_argument('--jitter', type=float, default=1e-10, help='regularizer that improves the stablity of matrix inversion')
    parser.add_argument('--n', type=int, default=-1, help='number of the representative samples')
    parser.add_argument('--sigma', type=float, default=1e-2, help='the noise level of the signal')
    args = parser.parse_args()

    main(args.fmat, args.y, args.prefix, args.test, args.jitter, args.n, args.sigma)
