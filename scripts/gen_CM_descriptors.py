#!/usr/bin/python3
import argparse
import os
import sys

import numpy as np
from ase.io import read, write
from dscribe.descriptors import CoulombMatrix


def main(fxyz, dictxyz, prefix, output, max_atoms, stride):
    """

    Generate the SOAP descriptors.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    dictxyz: string giving location of xyz file that is used as a dictionary
    prefix: string giving the filename prefix
    output: [xyz]: append the representations to extended xyz file; [mat] output as a standlone matrix
    max_atoms: int: Max number of atoms in the Coulomb Matrix
    stride: compute descriptor each X frames
    """

    fframes = []
    dictframes = []

    # read frames
    if fxyz != 'none':
        fframes = read(fxyz, slice(0, None, stride))
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
        frame.set_pbc([False, False, False])
    global_species = np.unique(global_species)
    print("a total of", nframes, "frames, with elements: ", global_species)

    rep_atomic = CoulombMatrix(max_atoms)
    foutput = prefix + "-max_atoms" + str(max_atoms)
    desc_name = "CM" + "-max_atoms" + str(max_atoms)

    # prepare for the output
    if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
    if os.path.isfile(foutput + ".desc"): os.rename(foutput + ".desc", "bck." + foutput + ".desc")

    for i, frame in enumerate(frames):
        fnow = rep_atomic.create(frame, n_jobs=8)

        frame.info[desc_name] = fnow
        # save
        if output == 'matrix':
            with open(foutput + ".desc", "ab") as f:
                np.savetxt(f, frame.info[desc_name][None])
                np.savetxt(fatomic, fnow)
        elif output == 'xyz':
            # output per-atom info
            # write xyze
            write(foutput + ".xyz", frame, append=True)
        else:
            raise ValueError('Cannot find the output format')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-fdict', type=str, default='none', help='Location of xyz file '
                                                                 'that is used for a dictionary')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--output', type=str, default='xyz', help='The format for output files ([xyz], [matrix])')
    parser.add_argument('--max_atoms', type=int, default=30,
                        help='Max number of atoms in the Coulomb Matrix')
    parser.add_argument('--stride', type=int, default=1,
                        help='Read in the xyz trajectory with X stide. Default: read/compute all frames')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fxyz, args.fdict, args.prefix, args.output, args.max_atoms, args.stride)
