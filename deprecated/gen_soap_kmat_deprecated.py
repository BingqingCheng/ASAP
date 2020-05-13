#!/usr/bin/python3
"""
python3 gen_soap_kmat.py --fxyz *.xyz --dict *.xyz --prefix $prefix
--rcut $rcut --n $nmax --l $lmax --g $g --periodic True/False --plot True/False
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel

from asaplib.io import str2bool


def main(fxyz, dictxyz, prefix, soap_rcut, soap_g, soap_n, soap_l, soap_periodic, matrix_plot):
    """

    Generate the SOAP kernel matrix.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    dictxyz: string giving location of xyz file that is used as a dictionary
    prefix: string giving the filename prefix
    soap_rcut: float giving the cutoff radius, default value is 3.0
    soap_g: float giving the atom width
    soap_n: int giving the maximum radial label
    soap_l: int giving the maximum angular label. Must be less than or equal to 9
    soap_periodic: string (True or False) indicating whether the system is periodic
    matrix_plot: string (True or False) indicating whether a plot of the kernel matrix
                 is to be generated
    """

    soap_periodic = bool(soap_periodic)
    fframes = []
    dictframes = []

    # read frames
    if fxyz != 'none':
        fframes = read(fxyz, ':')
        nfframes = len(fframes)
        print("read xyz file:", fxyz, ", a total of", nfframes, "frames")
    # read frames in the dictionary
    if dictxyz != 'none':
        dictframes = read(dictxyz, ':')
        ndictframes = len(dictframes)
        print("read xyz file used for a dictionary:", dictxyz, ", a total of",
              ndictframes, "frames")

    frames = dictframes + fframes
    nframes = len(frames)
    global_species = []
    for frame in frames:
        global_species.extend(frame.get_atomic_numbers())
        if not soap_periodic:
            frame.set_pbc([False, False, False])
    global_species = np.unique(global_species)
    print("a total of", nframes, "frames, with elements: ", global_species)

    if nframes > 1:
        # set up the soap descriptors
        soap_desc = SOAP(species=global_species, rcut=soap_rcut, nmax=soap_n, lmax=soap_l,
                         sigma=soap_g, crossover=False, average=True, periodic=soap_periodic)
    else:
        # if only one frame we compute the kernel matrix (kmat) between the atomic environments
        # within this frame
        soap_desc = SOAP(species=global_species, rcut=soap_rcut, nmax=soap_n, lmax=soap_l,
                         sigma=soap_g, crossover=False, average=False, periodic=soap_periodic)

    # compute soap fingerprints
    fall = soap_desc.create(frames, n_jobs=8)

    # compute kmat
    fshape = np.shape(fall)
    re = AverageKernel(metric="linear")

    kNN = re.create(fall.reshape((fshape[0], 1, fshape[1])))

    # save
    np.savetxt(prefix + "-n" + str(soap_n) + "-l" + str(soap_l) + "-c" + str(soap_rcut) + "-g" + str(soap_g) + ".kmat",
               kNN, fmt='%4.8f')

    # plot
    if matrix_plot:
        plt.matshow(kNN)
        plt.title('Kernel matrix: ' + prefix)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-fdict', type=str, default='none', help='Location of xyz file '
                                                                 'that is used for a dictionary')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--rcut', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--n', type=int, default=6, help='Maximum radial label')
    parser.add_argument('--l', type=int, default=6, help='Maximum angular label (<= 9)')
    parser.add_argument('--g', type=float, default=0.5, help='Atom width')
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=True,
                        help='Is the system periodic (True/False)?')
    parser.add_argument('--plot', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to plot the kernel matrix (True/False)?')
    args = parser.parse_args()

    main(args.fxyz, args.fdict, args.prefix, args.rcut, args.g, args.n, args.l, args.periodic, args.plot)
