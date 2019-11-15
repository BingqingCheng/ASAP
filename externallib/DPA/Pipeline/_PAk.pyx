# file '_PAk.pyx'. 
#
# Author: Maria d'Errico <mariaderr@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
cimport numpy as c_np
from math import log, sqrt, exp, lgamma, pi, pow
from scipy.optimize import minimize
import sys
from Pipeline import NR
cimport cython

DBL_MIN = sys.float_info.min
DBL_MAX = sys.float_info.max


def ratio_test(i, N, V1, V_dic, dim, distances, k_max, D_thr, indices):
    # Compute the volume of the dim-sphere with unitary radius
    cdef float Dk, vi, vj
    cdef int k, j 
    # Volumes are stored in V_dic dictionary to avoid double computations
    if i not in V_dic.keys():
        V_dic[i] = [-1]*k_max
        V_dic[i][0] = V1*exp(dim*log(distances[i][1]))
        V_dic[i][1] = V1*exp(dim*log(distances[i][2]))
        V_dic[i][2] = V1*exp(dim*log(distances[i][3]))
    k = 3 # Minimum number of neighbors required
    Dk = 0
    # Stop when the k+1-th neighbors has a significantly different density from point i 
    while (k<k_max and Dk<=D_thr):
        # Note: the k-th neighbor is at position k in the distances and indices arrays
        if i in V_dic.keys() and V_dic[i][k-1] != -1:
            vi = V_dic[i][k-1]
        elif i not in V_dic.keys():
            V_dic[i] = [-1]*k_max
            V_dic[i][0] = V1*exp(dim*log(distances[i][1]))
            V_dic[i][1] = V1*exp(dim*log(distances[i][2]))
            V_dic[i][2] = V1*exp(dim*log(distances[i][3]))
            V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
            vi = V_dic[i][k-1]
        else:
            V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
            vi = V_dic[i][k-1]
        # Check on the k+1-th neighbor of i
        j = indices[i][k+1]
        if j in V_dic.keys() and V_dic[j][k-1] != -1:
            vj = V_dic[j][k-1]
        elif j not in V_dic.keys():
            V_dic[j] = [-1]*k_max
            V_dic[j][0] = V1*exp(dim*log(distances[j][1]))
            V_dic[j][1] = V1*exp(dim*log(distances[j][2]))
            V_dic[j][2] = V1*exp(dim*log(distances[j][3]))
            V_dic[j][k-1] = V1*exp(dim*log(distances[j][k]))
            vj = V_dic[j][k-1]
        else:
            V_dic[j][k-1] = V1*exp(dim*log(distances[j][k]))
            vj = V_dic[j][k-1]

        Dk = -2.*k*(log(vi)+log(vj)-2.*log(vi+vj)+log(4.))
        k += 1
    V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
    return k, distances[i][k-1], V_dic


@cython.cdivision(True)
def get_densities(dim, distances, k_max, D_thr, indices):
    """Main function implementing the Pointwise Adaptive k-NN density estimator.

    Parameters
    ----------
    distances: array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points. The point is included. 
  
    indices : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points. The point is included.

    k_max : int, default=1000
        The maximum number of nearest-neighbors considered by the procedure that returns the
        largest number of neighbors ``\hat{k}`` for which the condition of constant density 
        holds, within a given level of confidence. If the number of points in the sample N is 
        less than the default value, k_max will be set automatically to the value ``N/2``.
    
    D_thr : float, default=23.92812698
        Set the level of confidence. The default value corresponds to a p-value of 
        ``10**{-6}`` for a ``\chiˆ2`` distribution with one degree of freedom.

    dim : int
        Intrinsic dimensionality of the sample.

    Results
    -------
    densities : array [n_samples]
        The logarithm of the density at each point.
    
    err_densities : array [n_samples]
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat : array [n_samples]
        The optimal number of neighbors for which the condition of constant density holds.

    dc : array [n_sample]
        The radius of the optimal neighborhood for each point.
    
    """
    cdef float V1 = exp(dim/2.*log(pi)-lgamma((dim+2)/2.))    
    cdef int N = distances.shape[0]
    cdef list k_hat = []
    #cdef c_np.ndarray[double, ndim=1] dc = np.array([])
    #cdef c_np.ndarray[double, ndim=1] densities = np.array([])
    #cdef c_np.ndarray[double, ndim=1] err_densities = np.array([])
    cdef list dc = []
    cdef list densities = []
    cdef list err_densities = []
    cdef dict V_dic = {}
    cdef int k, identical
    cdef double dc_i
    cdef c_np.ndarray[double, ndim=1] Vi 
    cdef float rho_min = DBL_MAX
    for i in range(0,N):
        k, dc_i, V_dic = ratio_test(i, N, V1, V_dic, dim, distances, k_max, D_thr, indices)
        k_hat.append(k-1)
        #dc = np.append(dc, dc_i)
        #densities = np.append(densities, log(k-1)-(log(V1)+dim*log(dc[i]))) 
        #err_densities = np.append(err_densities, sqrt((4.*(k-1)+2.)/((k-1)*(k-2))))
        dc.append(dc_i)
        densities.append(log(k-1)-(log(V1)+dim*log(dc[i]))) 
        err_densities.append(sqrt((4.*(k-1)+2.)/((k-1)*(k-2))))
        # Apply a correction to the density estimation if no neighbors are at the same distance from point i 
        densities[i] = NR.nrmaxl(densities[i], k_hat[i], V_dic[i], k_max)
        if densities[i] < rho_min:
            rho_min = densities[i]
    # Apply shift to have all densities as positive values 
    densities = [x-rho_min+1 for x in densities]
    
    return k_hat, dc, densities, err_densities

