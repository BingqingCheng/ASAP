#!/usr/bin/python3
import argparse
import json
import sys

from asaplib.hypers import gen_default_acsf_hyperparameters
from asaplib.io import str2bool

"""
Automatically generate the hyperparameters of ACSF descriptors for arbitrary elements and combinations.

## Example
The command 
gen_universal_acsf_hypers.py --Zs 5 32
will return length scales needed to define the ACSF descriptors for 
a system with boron (5) and germanium (32). 

"""


def main(Zs, sharpness, scalerange, verbose, outfile):
    hypers = gen_default_acsf_hyperparameters(Zs, scalerange, sharpness, verbose)

    # output
    if outfile == 'none':
        json.dump(hypers, sys.stdout)
        print("")
    else:
        with open(outfile, 'w') as jd:
            json.dump(hypers, jd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Zs", nargs="+", type=int, help="atomic numbers to calculate descriptors for", required=True)
    parser.add_argument("--sharpness", type=float,
                        help="sharpness factor.", default=1.0)
    parser.add_argument("--scalerange", type=float, help="scale the cutoff",
                        default=1.2)
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=True,
                        help="more descriptions of what has been done")
    parser.add_argument("--output", type=str, default='none', help="name of the output file")

    args = parser.parse_args()

    main(args.Zs, args.sharpness, args.scalerange, args.verbose, args.output)
