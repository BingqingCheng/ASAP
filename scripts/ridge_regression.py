#!/usr/bin/python3
"""
Python script for performing ridge regression. 
(with optional learning curve)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from asaplib.data import ASAPXYZ
from asaplib.fit import RidgeRegression
from asaplib.compressor import exponential_split, LCSplit, ShuffleSplit
from asaplib.fit import get_score
from asaplib.io import str2bool
from asaplib.plot import plot_styles

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
    if os.path.isfile(fmat):
        try:
            desc = np.genfromtxt(fmat, dtype=float)
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

    if len(y_all) != len(desc):
        raise ValueError('Length of the vector of properties is not the same as number of samples')

    # scale & center
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print(scaler.fit(desc))
        desc = scaler.transform(desc)  # normalizing the features
    # add bias
    desc_bias = np.ones((np.shape(desc)[0], np.shape(desc)[1] + 1))
    desc_bias[:, 1:] = desc
    print(np.shape(desc_bias))
    # train test split
    if test_ratio > 0:
        X_train, X_test, y_train, y_test = train_test_split(desc_bias, y_all, test_size=test_ratio, random_state=42)
    else:
        X_train = X_test = desc_bias
        y_train = y_test = y_all
    n_train = len(X_train)
    n_test = len(X_test)

    # TODO: add sparsification


    # if sigma is not set...
    if sigma < 0:
        sigma = 0.001 * np.std(y_train)

    rr = RidgeRegression(sigma)
    # fit the model
    rr.fit(X_train, y_train)

    fit_error = {}
    # get the predictions for train set
    y_pred = rr.predict(X_train)
    # compute the CV score for the dataset
    train_error = get_score(y_pred, y_train)
    print("train score: ", train_error)
    fit_error['train_error'] = train_error
    # get the predictions for test set
    y_pred_test = rr.predict(X_test)
    # compute the CV score for the dataset
    test_error = get_score(y_pred_test, y_test)
    print("test score: ", test_error)
    fit_error['test_error'] = test_error
    # dump to file
    import json
    with open("RR_train_test_errors_4_"+prefix+".json", 'w') as fp:
        json.dump(fit_error, fp)

    # learning curve
    # decide train sizes
    if lc_points > 1:
        train_sizes = exponential_split(20, n_train - n_test, lc_points)
        print("Learning curves using train sizes: ", train_sizes)
        lc_stats = lc_repeats * np.ones(lc_points, dtype=int)
        lc = LCSplit(ShuffleSplit, n_repeats=lc_stats, train_sizes=train_sizes, test_size=n_test, random_state=10)

        scores = {size: [] for size in train_sizes}
        for lctrain, lctest in lc.split(y_train):
            Ntrain = len(lctrain)
            lc_X_train = X_train[lctrain, :]
            lc_y_train = y_train[lctrain]
            lc_X_test = X_test
            lc_y_test = y_test
            rr.fit(lc_X_train, lc_y_train)
            lc_y_pred = rr.predict(lc_X_test)
            scores[Ntrain].append(get_score(lc_y_pred, lc_y_test))

        sc_name = 'RMSE'
        Ntrains = []
        avg_scores = []
        avg_scores_error = []
        for Ntrain, score in scores.items():
            avg = 0.
            var = 0.
            for sc in score:
                avg += sc[sc_name]
                var += sc[sc_name] ** 2.
            avg /= len(score)
            var /= len(score);
            var -= avg ** 2.
            avg_scores.append(avg)
            avg_scores_error.append(np.sqrt(var))
            Ntrains.append(Ntrain)

        # output learning curve
        np.savetxt("RR_learning_curve_4_"+prefix+".dat",np.stack((Ntrains,avg_scores,avg_scores_error), axis=-1))

    plot_styles.set_nice_font()
    
    if lc_points > 1:
        fig = plt.figure(figsize=(8 * 2.1, 8))
        ax = fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    ax.plot(y_train, y_pred, 'b.', label='train')
    ax.plot(y_test, y_pred_test, 'r.', label='test')
    ax.legend()
    ax.set_title('Ridge regression for: ' + fy)
    ax.set_xlabel('actual y')
    ax.set_ylabel('predicted y')

    if lc_points > 1:
        ax2 = fig.add_subplot(122)
        ax2.errorbar(Ntrains, avg_scores, yerr=avg_scores_error, linestyle='', uplims=True, lolims=True)
        ax2.set_title('Learning curve')
        ax2.set_xlabel('Number of training samples')
        ax2.set_ylabel('Test {}'.format(sc_name))
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    plt.show()
    fig.savefig('RR_4_' + prefix + '.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, default='ASAP_desc',
                        help='Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-fy', type=str, default='none', help='Location of the list of properties (N floats)')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--scale', type=str2bool, nargs='?', const=True, default=True,
                        help='Scale the coordinates (True/False). Scaling highly recommanded.')
    parser.add_argument('--test', type=float, default=0.05, help='the test ratio')
    parser.add_argument('--sigma', type=float, default=-1, help='the noise level of the signal. Also the regularizer that improves the stablity of matrix inversion.')
    parser.add_argument('--lcpoints', type=int, default=10, help='the number of points on the learning curve, <= 1 means no learning curve')
    parser.add_argument('--lcrepeats', type=int, default=8, help='the number of sub-samples to take when compute the learning curve')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.fy, args.prefix, args.scale, args.test, args.sigma, args.lcpoints, args.lcrepeats)
