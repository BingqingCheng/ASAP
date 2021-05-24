#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse

import spglib
from ase import Atoms as atom
from ase.io import read, write


def show_symmetry(symmetry):
    for i in range(symmetry['rotations'].shape[0]):
        print("  --------------- %4d ---------------" % (i + 1))
        rot = symmetry['rotations'][i]
        trans = symmetry['translations'][i]
        print("  rotation:")
        for x in rot:
            print("     [%2d %2d %2d]" % (x[0], x[1], x[2]))
        print("  translation:")
        print("     (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2]))


def show_lattice(lattice):
    print("Basis vectors:")
    for vec, axis in zip(lattice, ("a", "b", "c")):
        print("%s %10.5f %10.5f %10.5f" % (tuple(axis, ) + tuple(vec)))


def show_cell(lattice, positions, numbers):
    show_lattice(lattice)
    print("Atomic points:")
    for p, s in zip(positions, numbers):
        print("%2d %10.5f %10.5f %10.5f" % ((s,) + tuple(p)))


def main(fxyz, prefix, verbose, precision):
    # read frames
    if fxyz != 'none':
        frames = read(fxyz, ':')
        nframes = len(frames)
        print("read xyz file:", fxyz, ", a total of", nframes, "frames")

    standardized_frames = []

    for frame in frames:
        space_now = spglib.get_spacegroup(frame, symprec=precision)  # spglib.get_symmetry(frame, symprec=1e-1))
        print(space_now)
        lattice, scaled_positions, numbers = spglib.standardize_cell(frame,
                                                                     to_primitive=1,
                                                                     no_idealize=1,
                                                                     symprec=precision)
        if verbose:
            show_cell(lattice, scaled_positions, numbers)
        # output
        frtemp = atom(numbers=numbers, cell=lattice, scaled_positions=scaled_positions, pbc=frame.get_pbc())
        frtemp.info['space_group'] = space_now
        standardized_frames.append(frtemp)

    write(prefix + '-standardized.xyz', standardized_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('--prefix', type=str, default='output', help='Filename prefix')
    parser.add_argument('--verbose', type=bool, default=False, help='Screen output cell information [True/False]')
    parser.add_argument('--precision', type=float, default=1e-1, help='Precision used for space group finding')
    args = parser.parse_args()

    main(args.fxyz, args.prefix, args.verbose, args.precision)
