#!/usr/bin/python3
import argparse
import os
import sys

import numpy as np
from ase.io import read, write
from dscribe.descriptors import SOAP

from asaplib.hypers import gen_default_soap_hyperparameters
from asaplib.io import str2bool, NpEncoder


def main(fxyz, dictxyz, prefix, output, peratom, fsoap_param, soap_rcut, soap_g, soap_n, soap_l, soap_periodic, stride):
    """

    Generate the SOAP descriptors.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    dictxyz: string giving location of xyz file that is used as a dictionary
    prefix: string giving the filename prefix
    output: [xyz]: append the SOAP descriptors to extended xyz file; [mat] output as a standlone matrix
    fsoap_param: use (possibly multiple sets) of SOAP descriptors using parameters specified in fsoap_param file (json format)
    soap_rcut: float giving the cutoff radius, default value is 3.0
    soap_g: float giving the atom width
    soap_n: int giving the maximum radial label
    soap_l: int giving the maximum angular label. Must be less than or equal to 9
    soap_periodic: string (True or False) indicating whether the system is periodic
    stride: compute descriptor each X frames
    """
    soap_periodic = bool(soap_periodic)
    peratom = bool(peratom)
    fframes = []
    dictframes = []

    # read frames
    if fxyz != 'none':
        fframes = read(fxyz, slice(0, None, stride))
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
        if not soap_periodic:
            frame.set_pbc([False, False, False])
    global_species = np.unique(global_species)
    print("a total of", nframes, "frames, with elements: ", global_species)

    if os.path.isfile(fsoap_param) or fsoap_param == 'smart' or fsoap_param == 'Smart' or fsoap_param == 'SMART':
        import json
        # load the parameter from json file
        if os.path.isfile(fsoap_param):
            try:
                with open(fsoap_param, 'r') as soapfile:
                    soap_js = json.load(soapfile)
            except:
                raise IOError('Cannot load the json file for soap parameters')

        # use the default parameters
        elif fsoap_param == 'smart' or fsoap_param == 'Smart' or fsoap_param == 'SMART':
            soap_js = gen_default_soap_hyperparameters(list(global_species))
            print(soap_js)
            with open('smart-soap-parameters', 'w') as jd:
                json.dump(soap_js, jd, cls=NpEncoder)

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

    elif fsoap_param == 'none':
        soap_desc_atomic = [SOAP(species=global_species, rcut=soap_rcut, nmax=soap_n, lmax=soap_l,
                                 sigma=soap_g, rbf="gto", crossover=False, average=False, periodic=soap_periodic)]
        foutput = prefix + "-n" + str(soap_n) + "-l" + str(soap_l) + "-c" + str(soap_rcut) + "-g" + str(soap_g)
        desc_name = "SOAP" + "-n" + str(soap_n) + "-l" + str(soap_l) + "-c" + str(soap_rcut) + "-g" + str(soap_g)

    # prepare for the output
    if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
    if os.path.isfile(foutput + ".desc"): os.rename(foutput + ".desc", "bck." + foutput + ".desc")

    for i, frame in enumerate(frames):
        fnow = soap_desc_atomic[0].create(frame, n_jobs=8)

        for soap_desc_atomic_now in soap_desc_atomic[1:]:
            fnow = np.append(fnow, soap_desc_atomic_now.create(frame, n_jobs=8), axis=1)

        # average over all atomic environments inside the system

        frame.info[desc_name] = fnow.mean(axis=0)

        # save
        if output == 'matrix':
            with open(foutput + ".desc", "ab") as f:
                np.savetxt(f, frame.info[desc_name][None])
            if peratom or nframes == 1:
                with open(foutput + ".atomic-desc", "ab") as fatomic:
                    np.savetxt(fatomic, fnow)
        elif output == 'xyz':
            # output per-atom info
            if peratom:
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
    parser.add_argument('--peratom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom descriptors for multiple frames (True/False)?')
    parser.add_argument('-param_path', type=str, default='none',
                        help='Specify the hyper parameters using a json file. You can set it to "smart" to try out our universal SOAP parameters.')
    parser.add_argument('--rcut', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--n', type=int, default=6, help='Maximum radial label')
    parser.add_argument('--l', type=int, default=6, help='Maximum angular label (<= 9)')
    parser.add_argument('--g', type=float, default=0.5, help='Atom width')
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=True,
                        help='Is the system periodic (True/False)?')
    parser.add_argument('--stride', type=int, default=1,
                        help='Read in the xyz trajectory with X stide. Default: read/compute all frames')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fxyz, args.fdict, args.prefix, args.output, args.peratom, args.param_path, args.rcut, args.g, args.n,
         args.l, args.periodic, args.stride)
