#!/usr/bin/python3
"""
python3 gen_rvsoap_kmat.py --fxyz *.xyz
--rcut $rcut --g $g --periodic True/False --plot True/False

Only works for a single frame!!!
"""

import argparse
import matplotlib.pyplot as plt
import sys
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
from scipy.spatial.distance import cdist

from asaplib.io import str2bool

def main(fxyz, prefix, cutoff, sigma, matrix_plot):

    sigma2 = sigma**2.

    # Get the ASE traj from xyz
    frame = read(fxyz, index=0, format='extxyz')
    natoms = len(frame.get_positions())

    # build neighborlist
    nl = NeighborList([cutoff]*natoms, skin=0., sorted=False, self_interaction=False,
                 bothways=True)    
    nl.update(frame)

    # compute displacements r_ij
    rij = {}
    for central_atom in range(natoms):
        indices, offsets = nl.get_neighbors(central_atom)
        displacements = []
        for i, offset in zip(indices, offsets):
           displacements.append(frame.positions[i] + np.dot(offset, frame.get_cell()) - frame.positions[central_atom] )
        rij[central_atom] = np.array(displacements)

    kmat = np.zeros((natoms,natoms),float)
    for i in range(natoms):
        for j in range(i):
            ddm = cdist(rij[i],rij[j])
            #print i, j, ddm
        kmat[i,j] = kmat[j,i] = np.sum( np.exp( -np.square(ddm) / sigma2 ))
        kmat[i,i] = len(rij[i])

    np.savetxt(prefix+"-rvSOAP"+"-r-"+str(cutoff)+"-g-"+str(sigma)+(".kmat"), kmat, fmt='%.4e')

    # plot
    if matrix_plot:
        plt.matshow(kmat)
        plt.title('Kernel matrix: ' + prefix)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file, only works for the first frame')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--rcut', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--g', type=float, default=0.5, help='Atom width')
    parser.add_argument('--plot', type=str2bool, nargs='?', const=True, default=True,
                        help='Do you want to plot the kernel matrix (True/False)?')
    args = parser.parse_args()

    main(args.fxyz, args.prefix, args.rcut, args.g, args.plot)
