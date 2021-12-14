#!/usr/bin/python3
import argparse
import os

from asaplib.data import ASAPXYZ


def main(fxyz, prefix):
    """

    Test if computing descriptors is working.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    prefix: string giving the filename prefix
    """

    # read frames
    asapxyz = ASAPXYZ(fxyz, 1, False)  # not periodic

    peratom = True
    tag = 'test'

    soap_js = {'soap1': {'type': 'SOAP',
                         'cutoff': 2.0,
                         'n': 2, 'l': 2,
                         'atom_gaussian_width': 0.2,
                         'rbf': 'gto', 'crossover': False}}

    acsf_js = {'acsf1': {'type': 'ACSF',
                         'cutoff': 2.0,
                         'g2_params': [[1, 1], [1, 2], [1, 3]],
                         'g4_params': [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]}}

    k2_js = {'lmbtr-k2': {'type': 'LMBTR_K2',
                          'k2': {
                              "geometry": {"function": "distance"},
                              "grid": {"min": 0, "max": 2, "n": 10, "sigma": 0.1},
                              "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3, "threshold": 1e-2}},
                          'periodic': False,
                          'normalization': "l2_each"}}

    kernel_js = {}
    kernel_js['k1'] = {'reducer_type': 'moment_average',
                       'zeta': 2,
                       'element_wise': False}
    kernel_js['k2'] = {'reducer_type': 'sum',
                       'element_wise': True}

    desc_spec_js = {'test_cm': {'type': "CM"},
                    'test_soap': {'atomic_descriptor': soap_js, 'reducer_function': kernel_js},
                    'test_acsf': {'atomic_descriptor': acsf_js, 'reducer_function': kernel_js},
                    'test_k2': {'atomic_descriptor': k2_js, 'reducer_function': kernel_js}}

    # compute the descripitors
    asapxyz.compute_global_descriptors(desc_spec_js, [], peratom, tag)

    asapxyz.write_computed_descriptors(prefix, ['test_cm', 'test_soap'], [0])

    asapxyz.write(prefix)
    asapxyz.save_state(tag)


def test_gen(tmpdir):
    """Test the generation using pytest"""
    inp_file = os.path.join(os.path.split(__file__)[0], 'small_molecules-1000.xyz')
    main(inp_file, str(tmpdir / 'ASAP-test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, default='small_molecules-1000.xyz', help='Location of xyz file')
    parser.add_argument('--prefix', type=str, default='ASAP-test', help='Filename prefix')
    args = parser.parse_args()
    main(args.fxyz, args.prefix)
