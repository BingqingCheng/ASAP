#!/usr/bin/python3
import argparse
import json
import sys

from asaplib.hypers import gen_default_soap_hyperparameters
from asaplib.io import str2bool

"""
Automatically generate the hyperparameters of SOAP descriptors for arbitrary elements and combinations.

## Example
The command 
gen_universal_soap_hypers.py --Zs 5 32
will return length scales needed to define the SOAP descriptors for 
a system with boron (5) and germanium (32). The output is
{"soap1": {"atom_gaussian_width": 0.78, "n": 6, "cutoff": 6.2, "l": 6}, "soap2": {"atom_gaussian_width": 0.4, "n": 6, "cutoff": 3.2, "l": 6}}

"""


def main(Zs, soap_n, soap_l, multisoap, sharpness, scalerange, verbose, outfile):
    hypers = gen_default_soap_hyperparameters(Zs, multisoap=multisoap, scalerange=scalerange, soap_n=soap_n, soap_l=soap_l, sharpness=sharpness, verbose=verbose)

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
    parser.add_argument("--n", type=int, help="nmax for SOAP descriptors", default=6)
    parser.add_argument("--l", type=int, help="lmax for SOAP descriptors", default=6)
    parser.add_argument("--multisoap", type=int, help="How many set of SOAP descriptors do you want to use?", default=2)
    parser.add_argument("--sharpness", type=float,
                        help="sharpness factor for atom_gaussian_width, scaled to heuristic for GAP", default=1.0)
    parser.add_argument("--scalerange", type=float, help="the range of the SOAP cutoffs, scaled to heuristic for GAP",
                        default=1.0)
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=False,
                        help="more descriptions of what has been done")
    parser.add_argument("--output", type=str, default='none', help="name of the output file")

    args = parser.parse_args()

    main(args.Zs, args.n, args.l, args.multisoap, args.sharpness, args.scalerange, args.verbose, args.output)
