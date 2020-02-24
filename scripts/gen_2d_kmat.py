#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from asaplib.io.extended_xyz import read, write

def main(
        xmat, 
        kmat,
        normalize=False,
        matrix_plot=False):
    # open
    with open(xmat, 'rb') as f:
        X = np.loadtxt(f)
    # compute
    K = X.dot(X.T)
    if normalize:
        z = 1./np.diagonal(K)**0.5
        K = K*np.outer(z,z)
    # store
    if kmat == 'none':
        kmat = xmat.replace('.desc', '.kmat')
    with open(kmat, 'wb') as f:
        np.savetxt(f, K)
    # plot?
    if matrix_plot:
        plt.matshow(K)
        plt.title('Kernel matrix: ' + kmat)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-xmat', type=str, required=True, help='Design matrix of shape n_samples x n_dim')
    parser.add_argument('-kmat', type=str, default='none', help='Output kernel matrix')
    parser.add_argument('-normalize', type=bool, default=True, help='Normalize kernel matrix')
    parser.add_argument('-matrix_plot', type=bool, default=False, help='Show matrix plot of kernel')
    args = parser.parse_args()

    main(
        xmat=args.xmat, 
        kmat=args.kmat, 
        normalize=args.normalize,
        matrix_plot=args.matrix_plot)
