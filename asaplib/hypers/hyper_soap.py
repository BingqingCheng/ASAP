"""
tools for generating hyperparameters for SOAP descriptors
"""

import numpy as np

from .univeral_length_scales import uni_length_scales, system_pair_bond_lengths, round_sigfigs

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

## Example

The command 
gen_default_soap_hyperparameters([5,32], soap_n=6, soap_l=6, multisoap=2, sharpness=1.0, scalerange=1.0, verbose=False)
will return length scales needed to define the SOAP descriptors for 
a system with boron (5) and germanium (32).
"""

def gen_default_soap_hyperparameters(Zs, soap_n=6, soap_l=6, multisoap=2, sharpness=1.0, scalerange=1.0, verbose=False):

    """
    Parameters
    ----------
    Zs : array-like, list of atomic species
    soap_n, soap_l: soap parameters
    multisoap: type=int, How many set of SOAP descriptors do you want to use? default=2
    sharpness: type=float, sharpness factor for atom_gaussian_width, scaled to heuristic for GAP, default=1.0
    range: type=float, the range of the SOAP cutoffs, scaled to heuristic for GAP, default=1.0
    verbose: type=bool, default=False, more descriptions of what has been done.
    """

    # check if the element is in the look up table
    #print(type(Zs))
    for Z in Zs:
        if str(Z) not in uni_length_scales:
            raise RuntimeError("key Z {} not present in length_scales table".format(Z))

    shortest_bond, longest_bond = system_pair_bond_lengths(Zs, uni_length_scales)
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
    hypers['soap'+str(num_soap)] = { 'species': Zs, 'cutoff' : float(round_sigfigs(r_cut,2)), 'n' : soap_n, 'l' : soap_l, 'atom_gaussian_width' : float(round_sigfigs(g_width,2)) } 

    if multisoap >= 2:
        # ratio between subsequent rcut values
        rcut_ratio = (rcut_max/rcut_min)**(1./(multisoap-1))
        while r_cut > rcut_min*1.01:
            num_soap += 1
            r_cut /= rcut_ratio
            g_width = r_cut/8.0/sharpness
            hypers['soap'+str(num_soap)] = { "species": Zs, 'cutoff' : float(round_sigfigs(r_cut,2)), 'n' : soap_n, 'l' : soap_l, 'atom_gaussian_width' : float(round_sigfigs(g_width,2)) } 

    return hypers


