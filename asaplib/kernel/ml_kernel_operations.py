"""
some operations for kernel and distance matrices
"""

import copy

import numpy as np


def normalizekernel(kernel):
    # first normalize the kernel matrix
    nkernel = copy.deepcopy(kernel)
    size = len(kernel)
    for i in range(size):
        nkernel[i, :] /= np.sqrt(kernel[i, i])
        nkernel[:, i] /= np.sqrt(kernel[i, i])
        nkernel[i, i] = 1.0
    return nkernel.clip(max=1)


def kerneltodis(kernel):
    # there can be many transformations between the k-matrix and the distance matrix
    # Here we use d_ij = sqrt(2 - 2*k_ij)
    # (k_ij is a normalized symmetric kernel)
    nk = normalizekernel(kernel)
    size = len(kernel)
    dis = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i - 1):
            dis[i, j] = dis[j, i] = np.sqrt(2. - 2. * nk[i, j])

    return dis.clip(min=0)


def kerneltodis_linear(kernel):
    # there can be many transformations between the k-matrix and the distance matrix
    # Here we use d_ij = 1-k_ij
    # (k_ij is a normalized symetric kernel)
    nk = normalizekernel(kernel)
    dis = 1. - nk
    return dis.clip(min=0)


def kerneltorho(kernel, delta):
    # we compute the "density" of the data from kernel matrix
    # delta is the charecteristic spread in similarity

    rho = np.zeros(len(kernel))
    allrhofromdis = np.exp((np.asmatrix(kernel) - 1.0) / delta)

    for i in range(len(allrhofromdis)):
        rho[i] = np.sum(allrhofromdis[i])

    return rho


def distorho_quick(dis, delta):
    # we compute the "density" of the data from distance matrix
    # the distance matrix can be computed such as
    # dis = kerneltodis(kernel)
    # delta is the charecteristic distance
    rho = np.zeros(len(dis))
    allrhofromdis = np.exp(dis * (-1. / delta))

    for i in range(len(allrhofromdis)):
        rho[i] += np.sum(allrhofromdis[i])

    return rho
