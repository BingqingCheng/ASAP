#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse

from ase.build import niggli_reduce, minimize_tilt
from ase.io import read, write


def main(fxyz, prefix, stride):
    # read frames
    if fxyz != 'none':
        frames = read(fxyz, ':')
        nframes = len(frames)
        print("read xyz file:", fxyz, ", a total of", nframes, "frames")

    for s in range(0, nframes, stride):
        frame = frames[s]
        niggli_reduce(frame)
        minimize_tilt(frame, order=range(0, 3), fold_atoms=True)
        write(prefix + '-' + str(s) + '.res', frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('--prefix', type=str, default='output', help='Filename prefix')
    parser.add_argument('--stride', type=int, default=1, help='output stride')
    args = parser.parse_args()

    main(args.fxyz, args.prefix, args.stride)
