#!/usr/bin/python3
"""
python3 gen_kmat.py --fxyz *.xyz --dict *.xyz --prefix $prefix 
--rcut $rcut --n $nmax --l $lmax --g $g --periodic True/False --plot True/False
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ase.io import read
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel

def main(fxyz, dictxyz, prefix, soap_rcut, soap_g, soap_n, soap_l, soap_periodic, matrix_plot):

    # set parameters
    fxyz = str(fxyz)
    dictxyz = str(dictxyz)
    soap_rcut = float(soap_rcut)
    soap_g = float(soap_g)
    soap_n = int(soap_n)
    soap_l = int(soap_l)
    soap_periodic = bool(soap_periodic)
    matrix_plot = bool(matrix_plot)
    fframes = []; dictframes = []

    # read frames
    if (fxyz != 'none'):
        fframes = read(fxyz,':')
        nfframes = len(fframes)
        print("read xyz file:", fxyz,", a total of",nfframes,"frames")
    # read frames in the dictionary
    if (dictxyz != 'none'):
        dictframes = read(dictxyz,':')
        ndictframes = len(dictframes)
        print("read xyz file used for a dictionary:", dictxyz,", a total of",ndictframes,"frames")

    frames = dictframes + fframes
    nframes = len(frames)
    global_species = []
    for frame in frames:
        global_species.extend(frame.get_atomic_numbers())
    global_species = np.unique(global_species)
    print("a total of", nframes,"frames, with elements: ", global_species)

    # set up the soap descriptors
    soap_desc = SOAP(species=global_species,
                 rcut=soap_rcut, nmax=soap_n, lmax=soap_l, sigma=soap_g,
                 crossover=False, average=True,periodic=soap_periodic)

    # compute soap finger prints
    fall = soap_desc.create(frames, n_jobs=8)

    # compute kmat
    fshape = np.shape(fall)
    re = AverageKernel(metric="linear")
    eva = re.create(fall.reshape((fshape[0], 1, fshape[1])))

    # save
    np.savetxt(prefix+"-n"+str(soap_n)+"-l"+str(soap_l)+"-c"+str(soap_rcut)+"-g"+str(soap_g)+".kmat", eva, fmt='%4.8f')

    # plot
    if (matrix_plot):
        plt.matshow(eva)
        plt.show()

    
##########################################################################################
##########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file')
    parser.add_argument('-fdict', type=str, default='none', help='Location of xyz file that is used for a dictionary')
    parser.add_argument('--prefix', type=str, default='', help='Filename prefix')
    parser.add_argument('--rcut', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--n', type=int, default=6, help='Maximum radial label')
    parser.add_argument('--l', type=int, default=6, help='Maximum angular label (<= 9)')
    parser.add_argument('--g', type=float, default=0.5, help='Atom width')
    parser.add_argument('--periodic', type=bool, default=True, help='Is the system periodic (True/False)?')
    parser.add_argument('--plot', type=bool, default=False, help='Do you want to plot the kernel matrix (True/False)?')
    args = parser.parse_args()

    main(args.fxyz, args.fdict, args.prefix, args.rcut, args.g, args.n, args.l, args.periodic, args.plot)
