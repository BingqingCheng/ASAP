"""
For geting numerical derivatives of a time dependent series
real or complex
input format
[[i, x_i]]
"""

import numpy as np


def gettxv(xt, dt=1):
    # assume the timestep is constant
    txv = np.zeros((len(xt) - 2, 3))
    mx = np.mean(xt[:, 1])
    # print mx
    for i in range(len(xt) - 2):
        txv[i, 0] = xt[i + 1, 0] * dt
        txv[i, 1] = xt[i + 1, 1] - mx
        txv[i, 2] = (xt[i + 2, 1] - xt[i, 1]) / ((xt[i + 2, 0] - xt[i, 0]) * dt)
    return txv


def getcomplextxv(xt, dt=1):
    # assume the timestep is constant
    txv = np.zeros((len(xt) - 2, 3), dtype=np.complex_)
    mx = np.mean(xt[:, 1]) + 1j * np.mean(xt[:, 2])
    # print mx
    for i in range(len(xt) - 2):
        txv[i, 0] = xt[i + 1, 0] * dt
        txv[i, 1] = xt[i + 1, 1] + 1j * xt[i + 1, 2] - mx
        txv[i, 2] = (xt[i + 2, 1] + 1j * xt[i + 2, 1] - xt[i, 1] - 1j * xt[i, 1]) / ((xt[i + 2, 0] - xt[i, 0]) * dt)
    return txv


def getfftxv(fxx):
    fdxx = np.zeros((len(fxx), 2), dtype=np.complex_)
    for i, f in enumerate(fxx):
        fdxx[i, 0] = fxx[i, 0]
        fdxx[i, 1] = 1j * fxx[i, 0] * fxx[i, 1]
    return fdxx
