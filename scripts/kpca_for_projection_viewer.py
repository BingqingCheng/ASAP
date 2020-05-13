#!python
"""
Minimal implementation of KernelPCA projection with SOAP vectors.

by Tamas K. Stenczel, partly based on pca.py, kpca.py and gen_soap_descriptors.py

Originally written for the projection viewer: https://github.com/chkunkel/projection_viewer
and its integration with ABCD: https://github.com/libatoms/abcd

Takes an xyz file and parameters for SOAP, computes descriptors, the kernel matrix and projects,
saving the coordinates into a the info or arrays of the xyz file. The SOAP vectors are not saved.
"""

import argparse
import sys

import ase.io
import numpy as np
from dscribe.descriptors import SOAP

from asaplib.io import str2bool
from asaplib.reducedim import KernelPCA


def main(xyz_filename, output_filename=None, cutoff=3., n_max=6, l_max=6, zeta=2.0,
         atom_sigma=0.5, pbc=True, peratom=False, pca_d=4, njobs=4):
    """KenelPCA on atomic or per-config SOAP vectors.

    `peratom=True` performs projection on SOAP vectors from arrays
    `peratom=False` performs projection on SOAP vectors from info

    Note:
    The previous ASAP pca.py was projecting from info anyways and with peratom=True transformed the SOAP vectors from
    arrays into that space as well.

    Parameters
    ----------
    xyz_filename
    prefix
    output_filename
    cutoff
    n_max
    l_max
    zeta
    atom_sigma
    pbc
    peratom
    pca_d
    njobs

    Returns
    -------

    """

    if output_filename is None:
        output_filename = "ASAP-pca-d{0}.xyz".format(str(pca_d))
    peratom = bool(peratom)

    # read the xyz file
    frames = ase.io.read(xyz_filename, ':')
    n_frames = len(frames)

    # species and PBC
    global_species = []
    for frame in frames:
        global_species.extend(frame.get_atomic_numbers())
        frame.set_pbc(pbc)
    global_species = np.unique(global_species)
    print("loaded xyz file: {fn}, with {n_frames} frames and elements: {ele}".format(fn=xyz_filename, n_frames=n_frames,
                                                                                     ele=global_species))

    # soap
    desc = SOAP(rcut=cutoff, nmax=n_max, lmax=l_max, sigma=atom_sigma, species=global_species, periodic=pbc,
                average=not peratom)

    # kernel
    soap_vectors = desc.create(frames, n_jobs=njobs)
    kNN = np.dot(soap_vectors, soap_vectors.T) ** zeta
    print("kernel matrix (shape: {}) calculated from vector set of shape {}".format(kNN.shape, soap_vectors.shape))

    # projection
    projection = KernelPCA(pca_d).fit_transform(kNN)

    # add coords to info/arrays
    if peratom:
        running_index = 0
        for at in frames:
            n_atoms = len(at)
            at.arrays['pca_coord'] = projection[running_index:running_index + n_atoms, :].copy()
            running_index += n_atoms
    else:
        for i, at in enumerate(frames):
            at.info['pca_coord'] = projection[i]

    # save
    ase.io.write(output_filename, frames, write_results=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # file operations related parameters
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('--output', type=str, default=None,
                        help='The name of the output file, default is ASAP-<fxyz>')

    # SOAP and kernel matrix parameters
    parser.add_argument('--d', type=int, default=4, help='number of the principle components to keep')
    parser.add_argument('--rcut', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--n', type=int, default=6, help='Maximum radial label')
    parser.add_argument('--l', type=int, default=6, help='Maximum angular label (<= 9)')
    parser.add_argument('--g', type=float, default=0.5, help='Atom width')
    parser.add_argument('--zeta', type=float, default=2.0, help='Power of the kernel covariance to take')
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=True,
                        help='Is the system periodic (True/False)?')
    parser.add_argument('--peratom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom pca coordinates (True/False)?')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    print(args)

    main(xyz_filename=args.fxyz,
         output_filename=args.output,
         pca_d=args.d,
         cutoff=args.rcut,
         n_max=args.n,
         l_max=args.l,
         zeta=args.zeta,
         atom_sigma=args.g,
         pbc=args.periodic,
         peratom=args.peratom)
