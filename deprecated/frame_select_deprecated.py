#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse
import os

import numpy as np

from asaplib.compressor import fps, CUR_deterministic
from asaplib.data import ASAPXYZ


def main(fxyz, fy, prefix, nkeep, algorithm, fmat, fkde, reweight_lambda):
    """
    Select frames from the supplied xyz file (fxyz) using one of the following algorithms:

    1. random: random selection
    2. fps: farthest point sampling selection. Need to supply a kernel matrix or descriptor matrix using -fmat
    3. sortmin/sortmax: select the frames with the largest/smallest value. Need to supply the vector of properties using
       -fy
    4. CUR decomposition
    5. Reweight according to the re-weighted distribution exp(-f/\lambda),
       where exp(-f) is the precomputed kernel density estimation of the original samples.

    Parameters
    ----------
    fxyz: Path to xyz file.
    fy: Path to the list of properties (N floats) or name of the tags in ase xyz file
    prefix: Filename prefix, default is ASAP
    nkeep: The number of representative samples to select
    algorithm: 'the algorithm for selecting frames ([random], [fps], [sort], [reweight])')
    fmat: Location of descriptor or kernel matrix file. Needed if you select [fps].
    You can use gen_kmat.py to compute it.
    reweight_lambda: select samples according to the re-weighted distribution exp(-f/\lambda),
              where exp(-f) is the kernel density estimation of the original samples.
    """

    # read the xyz file
    asapxyz = ASAPXYZ(fxyz)
    nframes = asapxyz.get_num_frames()

    if nkeep == 0:
        nkeep = nframes

    if fy != 'none':
        y_all = []
        try:
            y_all = np.genfromtxt(fy, dtype=float)
        except:
            y_all = asapxyz.get_property(fy)
        if len(y_all) != nframes:
            raise ValueError('Length of the vector of properties is not the same as number of samples')

    if algorithm == 'random' or algorithm == 'RANDOM':
        idx = np.asarray(range(nframes))
        sbs = np.random.choice(idx, nkeep, replace=False)

    elif algorithm == 'sortmax' or algorithm == 'sortmin':
        if fy == 'none':
            raise ValueError('must supply the vector of properties for sorting')

        idx = np.asarray(range(nframes))
        if algorithm == 'sortmax':
            sbs = [x for _, x in sorted(zip(y_all, idx))][:nkeep]
        elif algorithm == 'sortmin':
            sbs = [x for _, x in sorted(zip(y_all, idx))][nkeep:]

    elif algorithm == 'fps' or algorithm == 'FPS' or algorithm == 'cur' or algorithm == 'CUR':
        # for both algo we read in the descriptor matrix
        desc, _ = asapxyz.get_descriptors(fmat)
        if os.path.isfile(fmat):
            try:
                desc = np.genfromtxt(fmat, dtype=float)
            except:
                raise ValueError('Cannot load the kernel matrix')
        print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))

        # FPS
        if algorithm == 'fps' or algorithm == 'FPS':
            sbs, dmax_remain = fps(desc, nkeep, 0)
            print("Making farthest point sampling selection")
            np.savetxt(prefix + "-" + algorithm + "-n-" + str(nkeep) + '.error', dmax_remain, fmt='%4.8f',
                       header='the maximum remaining distance in FPS')
        # CUR decomposition
        if algorithm == 'cur' or algorithm == 'CUR':
            desc = np.asmatrix(desc)
            cov = np.dot(desc, desc.T)
            print("Making CUR selection")
            print("shape of the covariance matrix:", np.shape(cov))
            sbs, rcov_error = CUR_deterministic(cov, nkeep)
            np.savetxt(prefix + "-" + algorithm + "-n-" + str(nkeep) + '.error', rcov_error, fmt='%4.8f',
                       header='the remaining error of the covariance matrix')

    elif algorithm == 'reweight':
        if os.path.isfile(fkde):
            try:
                logkde = np.genfromtxt(fkde, dtype=float)[:, 1]
            except:
                raise IOError('Cannot load the (log of) kernel density for each sample')
            if len(logkde) != nframes: raise ValueError('mismatch of number of frames and kernel densities')
        else:
            raise ValueError('must suply the (log of) kernel density for each sample')

        new_kde = np.zeros(nframes)
        for i in range(nframes):
            new_kde[i] = np.exp(logkde[i] / reweight_lambda) / np.exp(logkde[i])
        # compute the normalization factor so we expect to select n samples in the end
        normalization = nkeep / np.sum(new_kde)
        new_kde *= normalization
        sbs = []
        randomchoice = np.random.rand(nframes)
        for i in range(nframes):
            if randomchoice[i] < new_kde[i]:
                sbs.append(i)
        algorithm = algorithm + "-lambda-" + str(reweight_lambda)
    # save
    selection = np.zeros(nframes, dtype=int)
    for i in sbs:
        selection[i] = 1
    np.savetxt(prefix + "-" + algorithm + "-n-" + str(nkeep) + '.index', selection, fmt='%d')
    if fy != 'none':
        np.savetxt(prefix + "-" + algorithm + "-n-" + str(nkeep) + '-' + fy, np.asarray(y_all)[sbs], fmt='%4.8f')
    asapxyz.write(prefix + "-" + algorithm + "-n-" + str(nkeep), sbs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-y', type=str, default='none',
                        help='Location of the list of properties (N floats) or name of the tags in ase xyz file')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--n', type=int, default=0, help='number of the representative samples to select')
    parser.add_argument('--algo', type=str, default='random',
                        help='the algorithm for selecting frames ([random], [fps], [cur], [sortmax], [sortmin],[reweight])')
    parser.add_argument('-fmat', type=str, required=False,
                        help='Location of descriptor or kernel matrix file. Needed if you select [fps]. You can use gen_kmat.py to compute it.')
    parser.add_argument('-fkde', type=str, required=False,
                        help='Location of the (log of) kernel density for each sample. Needed if you select [reweight]. You can use kernel_density_estimation.py to compute it.')
    parser.add_argument('--reweight_lambda', type=float, default=1,
                        help='select samples according to the re-weighted distribution exp(-\lambda*f)')
    args = parser.parse_args()

    main(args.fxyz, args.y, args.prefix, args.n, args.algo, args.fmat, args.fkde, args.reweight_lambda)
