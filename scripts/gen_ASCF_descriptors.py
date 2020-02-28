#!/usr/bin/python3
import argparse
import os
import sys
import json


import numpy as np
from ase.io import read, write
from dscribe.descriptors import ACSF

from asaplib.io import str2bool


def main(fxyz, dictxyz, prefix, output, per_atom, r_cut , config_path , periodic):
    """

    Generate the SOAP descriptors.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    dictxyz: string giving location of xyz file that is used as a dictionary
    prefix: string giving the filename prefix
    output: [xyz]: append the SOAP descriptors to extended xyz file; [mat] output as a standlone matrix
    fmultisoap: use multiple sets of SOAP descriptors using parameters specified in fmultisoap file (json format)
    soap_rcut: float giving the cutoff radius, default value is 3.0
    soap_g: float giving the atom width
    soap_n: int giving the maximum radial label
    soap_l: int giving the maximum angular label. Must be less than or equal to 9
    periodic: string (True or False) indicating whether the system is periodic
    """
    periodic = bool(periodic)
    per_atom = bool(per_atom)
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
        if not periodic:
            frame.set_pbc([False, False, False])
    global_species = np.unique(global_species)
    print("a total of", nframes, "frames, with elements: ", global_species)

    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
    except Exception:
        raise IOError('Cannot load the json file for soap parameters')
    rep_atomic = ACSF(r_cut,**config)
    foutput = prefix + "-n" + str(r_cut)
    desc_name = "ACSF" + "-n" + str(r_cut)


    # prepare for the output
    if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
    if os.path.isfile(foutput + ".desc"): os.rename(foutput + ".desc", "bck." + foutput + ".desc")

    for i, frame in enumerate(frames):
        fnow = rep_atomic.create(frame, n_jobs=8)

        frame.info[desc_name] = fnow.mean(axis=0)

        # save
        if output == 'matrix':
            with open(foutput + ".desc", "ab") as f:
                np.savetxt(f, frame.info[desc_name][None])
            if per_atom or nframes == 1:
                with open(foutput + ".atomic-desc", "ab") as fatomic:
                    np.savetxt(fatomic, fnow)
        elif output == 'xyz':
            # output per-atom info
            if per_atom:
                frame.new_array(desc_name, fnow)
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
    parser.add_argument('--per_atom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom descriptors for multiple frames (True/False)?')
    parser.add_argument('--rcut', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--n', type=int, default=6, help='Maximum radial label')
    parser.add_argument('--l', type=int, default=6, help='Maximum angular label (<= 9)')
    parser.add_argument('--g', type=float, default=0.5, help='Atom width')
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=True,
                        help='Is the system periodic (True/False)?')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fxyz, args.fdict, args.prefix, args.output, args.peratom, args.multisoap, args.rcut, args.g, args.n,
         args.l, args.periodic)
