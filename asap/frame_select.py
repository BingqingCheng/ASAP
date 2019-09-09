#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse

from ase.io import read, write
from asaplib.compressor import fps
import numpy as np


def main(fxyz, fy, prefix, nkeep, algorithm, fkmat):

    # read frames
    frames = read(fxyz, ':')
    nframes = len(frames)
    print("read xyz file:", fxyz, ", a total of", nframes, "frames")

    if fy != 'none':
        y_all = []
        try:
            y_all = np.genfromtxt(fy, dtype=float)
        except: 
            try:
                for frame in frames:
                    y_all.append(frame.info[fy])
            except: raise ValueError('Cannot load the property vector')
        if len(y_all) != nframes:
            raise ValueError('Length of the vector of properties is not the same as number of samples')

    if algorithm == 'random' or algorithm == 'RANDOM':
        idx = np.asarray(range(nframes))
        sbs = np.random.choice(idx, nkeep, replace =False)

    elif algorithm == 'fps' or algorithm == 'FPS':
        try:
            kNN = np.genfromtxt(fkmat, dtype=float)
        except: raise ValueError('Cannot load the kernel matrix')
        sbs, _ = fps(kNN, nkeep , 0)

    # save
    np.savetxt(prefix+"-"+algorithm+"-n-"+str(nkeep)+'.index', sbs, fmt='%d')
    for i in sbs:
        write(prefix+"-"+algorithm+"-n-"+str(nkeep)+'.xyz',frames[i], append=True)
    if fy != 'none':
        np.savetxt(prefix+"-"+algorithm+"-n-"+str(nkeep)+'-'+fy, np.asarray(y_all)[sbs], fmt='%4.8f')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-y', type=str, default='none', help='Location of the list of properties (N floats) or tags in ase xyz file')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--n', type=int, default=1, help='number of the representative samples to select')
    parser.add_argument('--algo', type=str, default='random', help='the algotithm for selecting frames ([random], [fps])')
    parser.add_argument('-kmat', type=str, required=False, help='Location of kernel matrix file. Needed if you select [fps]. You can use gen_kmat.py to compute it.')
    args = parser.parse_args()

    main(args.fxyz, args.y, args.prefix, args.n, args.algo, args.kmat)
