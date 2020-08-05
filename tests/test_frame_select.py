#!/usr/bin/python3

import os

import numpy as np

from asaplib.compressor import Sparsifier
from asaplib.data import ASAPXYZ


def main():
    """
    Select frames from the supplied xyz file (fxyz) using one of the following algorithms:

    1. random: random selection
    2. fps: farthest point sampling selection. Need to supply a kernel matrix or descriptor matrix using -fmat
    4. CUR decomposition

    Parameters
    ----------
    fxyz: Path to xyz file.
    fmat: Path to the design matrix or name of the tags in ase xyz file
    prefix: Filename prefix, default is ASAP
    nkeep: The number of representative samples to select
    algorithm: 'the algorithm for selecting frames ([random], [fps], [cur])')
    fmat: Location of descriptor or kernel matrix file. Needed if you select [fps] or [cur].
    """

    fxyz = os.path.join(os.path.split(__file__)[0], 'small_molecules-SOAP.xyz')
    fmat = ['SOAP-n4-l3-c1.9-g0.23']
    nkeep = 10
    prefix = "test-frame-select"

    # read the xyz file
    asapxyz = ASAPXYZ(fxyz)
    # for both algo we read in the descriptor matrix
    desc, _ = asapxyz.get_descriptors(fmat)
    print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))

    for algorithm in ['random', 'cur', 'fps']:
        sparsifier = Sparsifier(algorithm)
        sbs = sparsifier.sparsify(desc, nkeep)
        # save
        selection = np.zeros(asapxyz.get_num_frames(), dtype=int)
        for i in sbs:
            selection[i] = 1
        np.savetxt(prefix + "-" + algorithm + "-n-" + str(nkeep) + '.index', selection, fmt='%d')
        asapxyz.write(prefix + "-" + algorithm + "-n-" + str(nkeep), sbs)


def test_gen(tmpdir):
    """Test the generation using pytest"""
    main()


if __name__ == '__main__':
    main()
