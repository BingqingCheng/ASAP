"""
tools for doing PCA
e.g.
eva = np.genfromtxt(prefix+".desc")
proj = pca(eva,6)
"""

import numpy as np
import scipy.linalg as salg


def pca(desc, ndim=2):
    """
    Parameters
    ----------
    desc : array-like, shape=[n_descriptors, n_samples]
        Input points.
    ndim : number of the principle components to keep
    """
    
    print("a total of ", np.shape(desc), "column")
    # center columns by subtracting column means
    C_desc = centering(desc)

    # calculate covariance matrix of centered matrix
    V = np.cov(C_desc.T)
    print("computing covariance matrix with shape:", np.shape(V))

    print("  And now we build a projection ")
    eval, evec = salg.eigh(V, eigvals=(len(V)-ndim, len(V)-1))
    eval = np.flipud(eval)
    evec = np.fliplr(evec)

    pvec = evec.copy()
    for i in range(ndim):
        pvec[:,i] *= 1./np.sqrt(eval[i])
    print("Done, super quick. ")

    return pca_project(desc, pvec), pvec 


def pca_project(desc, pvec):
    return np.dot(desc, pvec)


def centering(desc):
    
    # calculate the mean of each column
    M_desc = np.mean(desc.T, axis=1)
    # center columns by subtracting column means
    C_desc = desc - M_desc
    return C_desc
