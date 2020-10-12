"""
tools for generating hyperparameters for ACSF descriptors
"""
import json
import numpy as np

from .univeral_length_scales import uni_length_scales, system_pair_bond_lengths, round_sigfigs
from ..io import NpEncoder

"""
Automatically generate the hyperparameters of ACSF descriptors for arbitrary elements and combinations.

## Heuristics:
  * Get the length scales of the system from
    * maximum bond length (from equilibrium bond length in lowest energy 2D or 3D structure)
    * minimal bond length (from shortest bond length of any equilibrium structure, including dimer)
  * Apply a scaling for these length scales
    * acsf cutoff = maximum bond length * 1.56 * scalerange
    * rmin = minimal bond length, distance in Angstrom to the first nearest neighbor.
         Eliminates the symmetry functions that investigate the space between 0 and rmin.
    * N = the number of each type of SFs. Determined by
          N = min(int(sharpness*(cutoff-rmin)/0.5), 5)
## Example
"""

def universal_acsf_hyper(global_species, facsf_param, dump=True, verbose=True):

    """
    format:
    acsf_js = {'acsf1': {'type': 'ACSF',
                        'cutoff': 2.0,
                        'g2_params': [[1, 1], [1, 2], [1, 3]],
                        'g4_params': [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]}}
    """
    
    if facsf_param == 'smart' or facsf_param == 'Smart' or facsf_param == 'SMART':
        acsf_js = gen_default_acsf_hyperparameters(list(global_species), scalerange=1.2, sharpness=1.0)
    elif facsf_param == 'minimal' or facsf_param == 'Minimal' or facsf_param == 'MINIMAL':
        acsf_js = gen_default_acsf_hyperparameters(list(global_species), scalerange=0.85, sharpness=1.0)
    elif facsf_param == 'longrange' or facsf_param == 'Longrange' or facsf_param == 'LONGRANGE':
        acsf_js = gen_default_acsf_hyperparameters(list(global_species),scalerange=1.8, sharpness=1.2)
    elif isinstance(facsf_param, float):
        acsf_js = gen_default_acsf_hyperparameters(list(global_species),scalerange=1.0, sharpness=1.0, cutoff=facsf_param)
    else:
        raise IOError('Did not specify acsf parameters. You can use [smart/minimal/longrange] or set an explicit cutoff.')
        
    if verbose:  
        print(acsf_js)
    if dump:
        with open('smart-acsf-parameters', 'w') as jd:
            json.dump(acsf_js, jd, cls=NpEncoder)
    return acsf_js

def gen_default_acsf_hyperparameters(Zs, scalerange=1.0, sharpness=1.0, verbose=False, cutoff=None):
    """
    Parameters
    ----------
    Zs : array-like, list of atomic species
    scalerange: type=float, scale the cutoffs of the SFs.
    sharpness: type=float, sharpness factor for atom_width, default=1.0,
                larger sharpness means more resolution, and more SFs will be generated.
    verbose: type=bool, default=False, more descriptions of what has been done.
    """

    # check if the element is in the look up table 
    for Z in Zs:
        if str(Z) not in uni_length_scales:
            raise RuntimeError("key Z {} not present in length_scales table".format(Z))

    shortest_bond, longest_bond = system_pair_bond_lengths(Zs, uni_length_scales)
    if verbose:
        print(Zs, "range of bond lengths", shortest_bond, longest_bond)
        
    # cutoffs & shortest length
    if cutoff is None:
        cutoff = max(float(round_sigfigs(longest_bond * 1.3 * float(scalerange), 2)),2.0)
    rmin = shortest_bond
    N = int(sharpness*(cutoff-rmin)/0.5)
    if verbose:
        print("Considering cutoff and rmin", cutoff, rmin)

    index = np.arange(N+1, dtype=float)
    shift_array = cutoff*(1./N)**(index/(len(index)-1))
    eta_array = 1./shift_array**2.

    _2_body_params = []
    for eta in eta_array:
        # G2 with no shift
        if 3*np.sqrt(1/eta) > rmin:
            _2_body_params.append([float(round_sigfigs(eta, 2)), 0.])
            if verbose:
                for fel in Zs:
                    for sel in Zs: 
                        print("symfunction_short %s 2 %s %.4f 0.000 %.3f" %(fel, sel, eta, cutoff))
    for i in range(len(shift_array)-1):
        # G2 with shift
        eta = 1./((shift_array[N-i] - shift_array[N-i-1])**2)
        if shift_array[N-i] + 3*np.sqrt(1/eta) > rmin:
            _2_body_params.append([float(round_sigfigs(eta,2)), float(round_sigfigs(shift_array[N-i],2))])
            if verbose: 
                for fel in Zs:
                    for sel in Zs:
                        print("symfunction_short %s 2 %s %.4f %.3f %.3f" %(fel, sel, eta, shift_array[N-i], cutoff))

    eta_array = np.logspace(-3,0,N//2)
    zeta_array = [1.000, 4.000, 16.000]
    
    _3_body_params = []
    for eta in eta_array:
        for zeta in zeta_array:
            if 3*np.sqrt(1/eta) > rmin:
                _3_body_params.append([float(round_sigfigs(eta,2)), float(round_sigfigs(zeta,2)), 1])
                _3_body_params.append([float(round_sigfigs(eta,2)), float(round_sigfigs(zeta,2)), -1])

    if verbose:
        for fel in Zs:
            ang_Zs = list(Zs)
            for sel in Zs:
                for tel in ang_Zs:
                    print("# symfunctions for type %s 3 %s %s" %(fel, sel, tel))
                    for eta in eta_array:
                        for zeta in zeta_array:
                            if 3*np.sqrt(1/eta) > rmin:
                                print("symfunction_short %s 3 %s %s %.4f  1.000 %.3f %.3f" %(fel, sel, tel, eta, zeta, cutoff))
                                print("symfunction_short %s 3 %s %s %.4f -1.000 %.3f %.3f" %(fel, sel, tel, eta, zeta, cutoff))
                            
    acsf_js = { 'acsf-'+str(cutoff):{'type': 'ACSF',
    'cutoff': cutoff,
    'g2_params': _2_body_params,
    'g4_params': _3_body_params }}

    return acsf_js
