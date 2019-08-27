#!/usr/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
#from copy import deepcopy
from asaplib.fit import KRRSparse
from asaplib.fit import get_score
from asaplib.compressor import fps, kernel_random_split


def main(fkmat, fy, prefix, test_ratio, jitter, n_sparse, sigma):

    # if it has been computed before we can simply load it
    try:
        k_all = np.genfromtxt(fkmat, dtype=float)
    except: raise ValueError('Cannot load the kernel matrix')
    print("loaded",fkmat)
    try:
        y_all = np.genfromtxt(fy, dtype=float)
    except: raise ValueError('Cannot load the property vector')
    if (len(y_all) != len(k_all)): 
        raise ValueError('Length of the vector of properties is not the same as number of samples')
    else:
        n_sample = len(k_all)

    # train test split
    if (test_ratio > 0 ):
        X_train, X_test, y_train, y_test = kernel_random_split(k_all, y_all, test_ratio)
    else:
        X_train = X_test = k_all
        y_train = y_test = y_all

    # sparsification
    if (n_sparse >= len(X_train)):
        print("the number of representative structure is too large, please select n < ", len(X_train))
    elif (n_sparse > 0):
        ifps, dfps = fps(X_train, n_sparse , 0)
        #print(ifps)
        kMM = X_train[:,ifps][ifps]
        kNM = X_train[:,ifps]
        kTM = X_test[:,ifps]
    else:
        print("it's usually better to use some sparsification")
        kMM = X_train
        kNM = X_train
        kTM = X_test
        
    delta = np.std(y_train)/(np.trace(kMM)/len(kMM))
    krr = KRRSparse(jitter, delta, sigma)
    # fit the model
    krr.fit(kMM,kNM,y_train)

    # get the predictions for train set
    y_pred = krr.predict(kNM)
    # compute the CV score for the dataset
    print("train score: ", get_score(y_pred,y_train))
    # get the predictions for test set
    y_pred_test = krr.predict(kTM)
    # compute the CV score for the dataset
    print("test score: ", get_score(y_pred_test,y_test))

    fig, ax = plt.subplots() 
    ax.plot(y_train, y_pred,'b.',label='train')
    ax.plot(y_test, y_pred_test,'r.',label='test')
    ax.legend()
    plt.title('Kernel ridge regression test for: '+fy)
    plt.xlabel('actual y')
    plt.ylabel('predicted y')
    plt.show()
    fig.savefig('KRR_4_'+prefix+'.png')

##########################################################################################
##########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-kmat', type=str, required=True, help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-y', type=str, default='none', help='Location of the list of properties (N floats)')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--test', type=float, default=0.0, help='the test ratio')
    parser.add_argument('--jitter', type=float, default=1e-10, help='regularizor that improves the stablity of matrix inversion')
    parser.add_argument('--n', type=int, default=-1, help='number of the representative samples')
    parser.add_argument('--sigma', type=float, default=1e-2, help='the noise level of the signal')
    args = parser.parse_args()

    main(args.kmat, args.y, args.prefix, args.test, args.jitter, args.n, args.sigma)


