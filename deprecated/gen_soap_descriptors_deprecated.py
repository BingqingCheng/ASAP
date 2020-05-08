#!/usr/bin/python3
import argparse
import os
import sys

import numpy as np
from dscribe.descriptors import SOAP

from asaplib.data import ASAPXYZ
from asaplib.hypers import universal_soap_hyper
from asaplib.io import str2bool
from asaplib.kernel import Atomic_2_Global_Descriptor_By_Species


def main(fxyz, dictxyz, prefix, output, peratom, fsoap_param, soap_rcut, soap_g, soap_n, soap_l, zeta_list, kernel_type, element_wise, soap_periodic, stride):
    """

    Generate the SOAP descriptors.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    prefix: string giving the filename prefix
    output: [xyz]: append the SOAP descriptors to extended xyz file; [mat] output as a standlone matrix
    fsoap_param: use (possibly multiple sets) of SOAP descriptors using parameters specified in fsoap_param file (json format)
    soap_rcut: float giving the cutoff radius, default value is 3.0
    soap_g: float giving the atom width
    soap_n: int giving the maximum radial label
    soap_l: int giving the maximum angular label. Must be less than or equal to 9
    zeta_list : get the global descriptor from atomic ones of zeta th power
    kernel_type: type of operations to get global descriptors from the atomic soap vectors
    elementwise: consider different species seperately when computing global descriptors from the atomic soap vectors
    soap_periodic: string (True or False) indicating whether the system is periodic
    stride: compute descriptor each X frames
    """

    # read frames
    asapxyz = ASAPXYZ(fxyz)

    if fsoap_param is not None:
        import json
        # load the parameter from json file
        if os.path.isfile(fsoap_param):
            try:
                with open(fsoap_param, 'r') as soapfile:
                    soap_js = json.load(soapfile)
            except:
                raise IOError('Cannot load the json file for soap parameters')

        # use the default parameters
        else: 
            soap_js = universal_soap_hyper(global_species, fsoap_param, dump=True)

        # make descriptors
        soap_desc_atomic = []
        for element in soap_js.keys():
            soap_param = soap_js[element]
            [species_now, cutoff_now, g_now, n_now, l_now] = [soap_param['species'], soap_param['cutoff'],
                                                              soap_param['atom_gaussian_width'], soap_param['n'],
                                                              soap_param['l']]
            soap_desc_atomic.append(SOAP(species=species_now, rcut=cutoff_now, nmax=n_now, lmax=l_now,
                                         sigma=g_now, rbf="gto", crossover=False, average=False,
                                         periodic=soap_periodic))

        foutput = prefix + "-soapparam" + '-' + fsoap_param
        desc_name = "SOAPPARAM" + '-' + fsoap_param

    else:
        soap_desc_atomic = [SOAP(species=global_species, rcut=soap_rcut, nmax=soap_n, lmax=soap_l,
                                 sigma=soap_g, rbf="gto", crossover=False, average=False, periodic=soap_periodic)]
        foutput = prefix + "-n" + str(soap_n) + "-l" + str(soap_l) + "-c" + str(soap_rcut) + "-g" + str(soap_g)
        desc_name = "SOAP" + "-n" + str(soap_n) + "-l" + str(soap_l) + "-c" + str(soap_rcut) + "-g" + str(soap_g)
    for i, frame in enumerate(frames):
        fnow = soap_desc_atomic[0].create(frame, n_jobs=8)

        for soap_desc_atomic_now in soap_desc_atomic[1:]:
            fnow = np.append(fnow, soap_desc_atomic_now.create(frame, n_jobs=8), axis=1)
            
        if kernel_type == 'average' and element_wise == False and len(zeta_list)==1 and zeta_list[0]==1:
            # this is the vanilla situation. We just take the average soap for all atoms
            frame.info[desc_name] = Atomic_2_Global_Descriptor_By_Species(fnow, [], [], kernel_type, zeta_list)
        elif element_wise == False:
            frame.info[desc_name+'-'+kernel_type] = Atomic_2_Global_Descriptor_By_Species(fnow, [], [], kernel_type, zeta_list)
        else:
            frame.info[desc_name+'-'+kernel_type+'-elementwise'] = Atomic_2_Global_Descriptor_By_Species(fnow, frame.get_atomic_numbers(), global_species, kernel_type, zeta_list)

        # save
        if output == 'matrix':
            asapxyz.write_descriptor_matrix(desc_name, desc_name)
            if peratom or nframes == 1:
                asapxyz.write_atomic_descriptor_matrix(desc_name, desc_name)
        elif output == 'xyz':
           asapxyz.write(foutput)
        else:
            raise ValueError('Cannot find the output format')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-fdict', type=str, default=None, help='Location of xyz file '
                                                                 'that is used for a dictionary')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--output', type=str, default='xyz', help='The format for output files ([xyz], [matrix])')
    parser.add_argument('--peratom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom descriptors for multiple frames (True/False)?')
    parser.add_argument('-param_path', type=str, default=None,
                        help='Specify the hyper parameters using a json file. You can set it to "smart","minimal", or "longrange" to try out our universal SOAP parameters.')
    parser.add_argument('--rcut', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--n', type=int, default=6, help='Maximum radial label')
    parser.add_argument('--l', type=int, default=6, help='Maximum angular label (<= 9)')
    parser.add_argument('--g', type=float, default=0.5, help='Atom width')
    parser.add_argument('--zeta', nargs='+', type=int, default=[1], 
                       help='a list of the moments to take when converting atomic descriptors to global ones. e.g. 1 2 3 4, default:1')
    parser.add_argument('--kernel', type=str, default='average', help='type of operations to get global descriptors from the atomic soap vectors [average], [sum]')
    parser.add_argument('--elementwise', type=str2bool, default=False, help='element-wise operation to get global descriptors from the atomic soap vectors')
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=True,
                        help='Is the system periodic (True/False)?')
    parser.add_argument('--stride', type=int, default=1,
                        help='Read in the xyz trajectory with X stide. Default: read/compute all frames')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fxyz, args.fdict, args.prefix, args.output, args.peratom, args.param_path, args.rcut, args.g, args.n,
         args.l, args.zeta, args.kernel, args.elementwise, args.periodic, args.stride)
