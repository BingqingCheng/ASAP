#!/usr/bin/python3
import argparse
import os
import sys
import json

from asaplib.io import str2bool

"""
Automatically generate the hyperparameters of SOAP descriptors for arbitrary elements and combinations.

## Heuristics:
  * Get the length scales of the system from
    * maximum bond length (from equilibrium bond length in lowest energy 2D or 3D structure)
    * minimal bond length (from shortest bond length of any equilibrium structure, including dimer)
  * Apply a scaling for these length scales
    * largest soap cutoff = maximum bond length * 2.5
    * smallest soap cutoff = minimal bond length * 2.0
    * Add other cutoffs in between if requires more sets of SOAP descriptors 
    * The atom sigma is the `cutoff / 8`, divided by an optional `sharpness` factor

## Depedency: uses '../data/auto_length_scales.json' to read in length scales.

## Example

The command 
gen_universal_soap_hypers.py --Zs 5 32
will return length scales needed to define the SOAP descriptors for 
a system with boron (5) and germanium (32). The output is
{"soap1": {"atom_gaussian_width": 0.78, "n": 6, "cutoff": 6.2, "l": 6}, "soap2": {"atom_gaussian_width": 0.4, "n": 6, "cutoff": 3.2, "l": 6}}

"""

def system_pair_bond_lengths(Zs, length_scales):

    # shortest bond in this composition
    shortest_bond = min([length_scales[str(Z)]["min_bond_len"][0] for Z in Zs])

    # longest bond in this composition
    longest_bond = max([length_scales[str(Z)]["bond_len"][0] for Z in Zs])

    return shortest_bond, longest_bond


def round_sigfigs(v, n_sig_figs):
    # https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    return '{:g}'.format(float('{:.{p}g}'.format(v, p=n_sig_figs)))

def main(Zs, soap_n, soap_l, length_scales, multisoap, sharpness, scalerange, verbose, outfile):

    # check if the element is in the look up table
    for Z in Zs:
        if str(Z) not in length_scales:
            raise RuntimeError("key Z {} not present in length_scales table".format(Z))

    shortest_bond, longest_bond = system_pair_bond_lengths(Zs, length_scales)
    if verbose:
        print(Zs, "range of bond lengths", shortest_bond, longest_bond)

    # factor between shortest bond and shortest cutoff threshold
    factor_inner = 2.0 * scalerange
    rcut_min = factor_inner*shortest_bond
    # factor between longest bond and longest cutoff threshold
    factor_outer = 2.5 * scalerange
    rcut_max = factor_outer*longest_bond
    if verbose:
        print("Considering minimum and maximum cutoff", rcut_min, rcut_max)

    hypers = {}
    num_soap = 1    
    # first soap cutoff is just the rcut_max
    r_cut = rcut_max
    g_width = r_cut/8.0/sharpness
    hypers['soap'+str(num_soap)] = { 'cutoff' : float(round_sigfigs(r_cut,2)), 'n' : soap_n, 'l' : soap_l, 'atom_gaussian_width' : float(round_sigfigs(g_width,2)) } 

    if multisoap >= 2:
        # ratio between subsequent rcut values
        rcut_ratio = (rcut_max/rcut_min)**(1./(multisoap-1))
        while r_cut > rcut_min*1.01:
            num_soap += 1
            r_cut /= rcut_ratio
            g_width = r_cut/8.0/sharpness
            hypers['soap'+str(num_soap)] = { 'cutoff' : float(round_sigfigs(r_cut,2)), 'n' : soap_n, 'l' : soap_l, 'atom_gaussian_width' : float(round_sigfigs(g_width,2)) } 

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
    parser.add_argument("--multisoap", type=float, help="How many set of SOAP descriptors do you want to use?", default=2)
    parser.add_argument("--sharpness", type=float, help="sharpness factor for atom_gaussian_width, scaled to heuristic for GAP", default=1.0)
    parser.add_argument("--range", type=float, help="the range of the SOAP cutoffs, scaled to heuristic for GAP", default=1.0)
    parser.add_argument("--length_scales_file", help="JSON file with length scales", default="../data/auto_length_scales.json")
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=False, help="more descriptions of what has been done")
    parser.add_argument("--output", type=str, default='none', help="name of the output file")

    args = parser.parse_args()

    length_scales = json.load(open(args.length_scales_file))

    main(args.Zs, args.n, args.l, length_scales, args.multisoap, args.sharpness, args.range, args.verbose, args.output)
