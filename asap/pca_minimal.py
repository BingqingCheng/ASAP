#!python3

import argparse
import sys

import numpy as np
import ase.io
from asaplib.io import str2bool
from asaplib.pca import pca


def main(desc_key, fxyz, peratom, scale, pca_d, keep_raw=False, output=None, prefix='ASAP'):
    """
    PCA on atomic or per-config SOAP vectors.

    `peratom=True` performs projection on SOAP vectors from arrays
    `peratom=False` performs projection on SOAP vectors from info

    Note:
    The previous ASAP pca.py was projecting from info anyways and with peratom=True transformed the SOAP vectors from
    arrays into that space as well.

    :param desc_key:
    :param fxyz:
    :param prefix:
    :param peratom:
    :param scale:
    :param pca_d:
    :param keep_raw:
    :param output:
    :return:
    """

    if output is None:
        output = prefix + "-pca-d" + str(pca_d) + '.xyz'
    peratom = bool(peratom)

    # read the xyz file
    frames = ase.io.read(fxyz, ':')
    n_frames = len(frames)
    print('load xyz file: ', fxyz, ', a total of ', str(n_frames), 'frames')

    # extract the descriptors from the file
    desc = []
    if n_frames == 1 and not peratom:
        raise RuntimeError('Per-config PCA not possible on a single frame')

    # retrieve the SOAP vectors --- both of these throw a ValueError if any are missing or are of wrong shape
    if peratom:
        desc = np.concatenate([a.get_array(desc_key) for a in frames])
    else:
        desc = np.row_stack([a.info[desc_key] for a in frames])

    # scale & center
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print('DEBUG: {}'.format(desc.shape))
        print(scaler.fit(desc))
        desc = scaler.transform(desc)  # normalizing the features

    # fit PCA
    proj, pvec = pca(desc, pca_d)
    # could do with sklearn as well
    # from sklearn.decomposition import PCA
    # pca_sklearn = PCA(n_components=4) # can set svd_solver
    # proj = pca_sklearn.fit_transform(desc)
    # pvec = pca_sklearn.components_

    # add coords to info/arrays
    if peratom:
        running_index = 0
        for at in frames:
            n_atoms = len(at)
            at.arrays['pca_coord'] = proj[running_index:running_index + n_atoms, :].copy()
            running_index += n_atoms

        if not keep_raw:
            for at in frames:
                del at.arrays[desc_key]
    else:
        for i, at in enumerate(frames):
            at.info['pca_coord'] = proj[i]

        if not keep_raw:
            for at in frames:
                del at.info[desc_key]

    # save
    ase.io.write(output, frames, write_results=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--desc-key', type=str, default='ASAP_desc',
                        help='Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_'
                             'descriptors.py to compute it.')
    parser.add_argument('--fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('--output', type=str, default='matrix', help='The format for output files ([xyz], [matrix])')
    parser.add_argument('--peratom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom pca coordinates (True/False)?')
    parser.add_argument('--scale', type=str2bool, nargs='?', const=True, default=True,
                        help='Scale the coordinates (True/False). Scaling highly recommanded.')
    parser.add_argument('-d', type=int, default=10, help='number of the principle components to keep')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    print(args)

    main(desc_key=args.desc_key,
         fxyz=args.fxyz,
         output=args.output,
         peratom=args.peratom,
         scale=args.scale,
         pca_d=args.d)
