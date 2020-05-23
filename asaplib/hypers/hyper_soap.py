"""
tools for generating hyperparameters for SOAP descriptors
"""
import json

from .univeral_length_scales import uni_length_scales, system_pair_bond_lengths, round_sigfigs
from ..io import NpEncoder

"""
Automatically generate the hyperparameters of SOAP descriptors for arbitrary elements and combinations.

## Heuristics:
  * Get the length scales of the system from
    * maximum bond length (from equilibrium bond length in lowest energy 2D or 3D structure)
    * minimal bond length (from shortest bond length of any equilibrium structure, including dimer)
  * Apply a scaling for these length scales
    * largest soap cutoff = maximum bond length * 1.3
    * smallest soap cutoff = minimal bond length * 1.3
    * Add other cutoffs in between if requires more sets of SOAP descriptors 
    * The atom sigma is the `cutoff / 8`, divided by an optional `sharpness` factor

## Example

The command 
gen_default_soap_hyperparameters([5,32], soap_n=6, soap_l=6, multisoap=2, sharpness=1.0, scalerange=1.0, verbose=False)
will return length scales needed to define the SOAP descriptors for 
a system with boron (5) and germanium (32).
"""

def universal_soap_hyper(global_species, fsoap_param, dump=True):

    if fsoap_param == 'smart' or fsoap_param == 'Smart' or fsoap_param == 'SMART':
        soap_js = gen_default_soap_hyperparameters(list(global_species), multisoap=2, scalerange=1.2, soap_n=8, soap_l=4, sharpness=1.0)
    elif fsoap_param == 'minimal' or fsoap_param == 'Minimal' or fsoap_param == 'MINIMAL':
        soap_js = gen_default_soap_hyperparameters(list(global_species), multisoap=1, scalerange=0.85, soap_n=4, soap_l=3, sharpness=1.0)
    elif fsoap_param == 'longrange' or fsoap_param == 'Longrange' or fsoap_param == 'LONGRANGE':
        soap_js = gen_default_soap_hyperparameters(list(global_species), multisoap=2, scalerange=1.8, soap_n=8, soap_l=4, sharpness=1.2)
    else:
        raise IOError('Did not specify soap parameters. You can use [smart/minimal/longrange].')
    print(soap_js)
    if dump:
        with open('smart-soap-parameters', 'w') as jd:
            json.dump(soap_js, jd, cls=NpEncoder)
    return soap_js

def gen_default_soap_hyperparameters(Zs, multisoap=2, scalerange=1.0, soap_n=8, soap_l=4, sharpness=1.0, verbose=False):
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
    # print(type(Zs))
    for Z in Zs:
        if str(Z) not in uni_length_scales:
            raise RuntimeError("key Z {} not present in length_scales table".format(Z))

    shortest_bond, longest_bond = system_pair_bond_lengths(Zs, uni_length_scales)
    if verbose:
        print(Zs, "range of bond lengths", shortest_bond, longest_bond)

    # factor between shortest bond and shortest cutoff threshold
    factor_inner = 1.3 * scalerange
    rcut_min = max(2.0, factor_inner * shortest_bond)
    # factor between longest bond and longest cutoff threshold
    factor_outer = 1.3 * scalerange
    rcut_max = max(rcut_min * 1.2, factor_outer * longest_bond)
    if verbose:
        print("Considering minimum and maximum cutoff", rcut_min, rcut_max)

    hypers = {}
    num_soap = 1
    # first soap cutoff is just the rcut_max
    r_cut = rcut_max
    g_width = r_cut / 8.0 / sharpness
    hypers['soap' + str(num_soap)] = {'type': 'SOAP',
                                      'species': Zs, 
                                      'cutoff': float(round_sigfigs(r_cut, 2)), 
                                      'n': soap_n, 'l': soap_l,
                                      'atom_gaussian_width': float(round_sigfigs(g_width, 2))}

    if multisoap >= 2:
        # ratio between subsequent rcut values
        rcut_ratio = (rcut_max / rcut_min) ** (1. / (multisoap - 1))
        while r_cut >= rcut_max * 0.99:
            num_soap += 1
            r_cut /= rcut_ratio
            g_width = r_cut / 8.0 / sharpness
            hypers['soap' + str(num_soap)] = {'type': 'SOAP',
                                              "species": Zs, 
                                              'cutoff': float(round_sigfigs(r_cut, 2)), 
                                              'n': soap_n,
                                              'l': soap_l, 
                                              'atom_gaussian_width': float(round_sigfigs(g_width, 2))}

    return hypers
