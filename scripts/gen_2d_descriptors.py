#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import asaplib.io
from asaplib.io.extended_xyz import read, write
from asaplib.io import str2bool
try:
    import rdkit.Chem as rchem
    from rdkit.Chem import AllChem as rallchem
except ImportError:
    raise ImportError("Failed to import rdkit.Chem: Install via 'conda -c rdkit rdkit'.")

def main(
        fxyz, 
        dictxyz, 
        prefix, 
        output, 
        smiles_key='canonical_smiles',
        normalize=True,
        fp_radius=3,
        fp_length=2048,
        peratom=False,
        periodic=False):
    """
    Generate circular fingerprints for extended xyz file objects

    Parameters
    ----------
    fp_radius: connectivity cutoff
    fp_length: bit length after hash
    normalize: apply 2-norm to fps
    smiles_key: smiles key from key-value pairs in ext-xyz comment line
    """
    peratom = bool(peratom)
    fframes = []
    dictframes = []

    # read frames
    if fxyz != 'none':
        fframes = read(fxyz, ':')
        nfframes = len(fframes)
        print("read xyz file:", fxyz, ", a total of", nfframes, "frames")
    # read frames in the dictionary
    if dictxyz != 'none':
        dictframes = read(dictxyz, ':')
        ndictframes = len(dictframes)
        print("read xyz file used for a dictionary:", dictxyz, ", a total of",
              ndictframes, "frames")
    # join frames
    frames = dictframes + fframes
    nframes = len(frames)
    if not periodic:
        for frame in frames:
            frame.set_pbc([False, False, False])

    # prepare for the output
    desc_name = "_rchem2d_radius%d_bits%d_norm%d" % (fp_radius, fp_length, 1 if normalize else 0)
    foutput = prefix + desc_name
    print(foutput)
    if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
    if os.path.isfile(foutput + ".desc"): os.rename(foutput + ".desc", "bck." + foutput + ".desc")

    # descriptors
    smiles = [ c.info["canonical_smiles"] for c in frames ]
    mols = [ rchem.MolFromSmiles(s) for s in smiles ]
    fps = [ rallchem.GetMorganFingerprintAsBitVect(
        mol, radius=fp_radius, nBits=fp_length) for mol in mols ]
    fps = np.array(fps, dtype='float64')
    if normalize:
        z = 1./(np.sum(fps**2, axis=1)+1e-10)**0.5
        fps = (fps.T*z).T

    # save
    if output == 'matrix':
        with open(foutput + ".desc", "wb") as f:
            np.savetxt(f, fps)
    elif output == 'xyz':
        raise IOError("Output type 'xyz' not supported for 2d descriptors")
    else:
        raise ValueError('No such output format: "%s"' % output)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-fdict', type=str, default='none', help='Location of xyz file '
                                                                 'that is used for a dictionary')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--output', type=str, default='matrix', help='The format for output files ([xyz], [matrix])')
    parser.add_argument('--peratom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom descriptors for multiple frames (True/False)?')
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=True,
                        help='Is the system periodic (True/False)?')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fxyz, args.fdict, args.prefix, args.output, 
        peratom=args.peratom, periodic=args.periodic, 
        normalize=True,
        fp_radius=3, 
        fp_length=2048)

