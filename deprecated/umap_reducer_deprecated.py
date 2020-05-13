#!/usr/bin/python3
"""
script for applying UMAP to a precomputed design matrix. See: https://arxiv.org/abs/1802.03426 and
https://umap-learn.readthedocs.io/en/latest/index.html
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap

from asaplib.data import ASAPXYZ
from asaplib.io import str2bool
from asaplib.plot import Plotters, set_color_function


def main(fmat, fxyz, ftags, fcolor, colorscol, prefix, output, peratom, keepraw, scale, umap_d, pc1, pc2,
        projectatomic, plotatomic, adtext):
    """

    Parameters
    ----------
    fmat: Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.
    fxyz: Location of xyz file for reading the properties.
    ftags: Location of tags for the first M samples. Plot the tags on the umap.
    fcolor: Location of a file or name of the tags in ase xyz file. It should contain properties for all samples (N floats) used to color the scatterplot'
    colorscol: The column number of the properties used for the coloring. Starts from 0.
    prefix: Filename prefix, default is ASAP
    output: The format for output files ([xyz], [matrix]). Default is xyz.
    peratom: Whether to output per atom t-SNE coordinates (True/False)
    keepraw: Whether to keep the high dimensional descriptor when output is an xyz file (True/False)
    scale: Scale the coordinates (True/False). Scaling highly recommanded.
    umap_d: Dimension of the embedded space.
    dim1: Plot the projection along which principle axes
    dim2: Plot the projection along which principle axes
    projectatomic: build the projection using the (big) atomic descriptor matrix
    plotatomic: Plot the PCA coordinates of all atomic environments (True/False)
    adtext: Whether to adjust the texts (True/False)

    Returns
    -------

    """

    foutput = prefix + "-pca-d" + str(umap_d)
    use_atomic_desc = (peratom or plotatomic or projectatomic)

    # try to read the xyz file
    if fxyz != 'none':
        asapxyz = ASAPXYZ(fxyz)
        desc, desc_atomic = asapxyz.get_descriptors(fmat, use_atomic_desc)
        if projectatomic: desc = desc_atomic.copy()
    else:
        asapxyz = None
        print("Did not provide the xyz file. We can only output descriptor matrix.")
        output = 'matrix'
    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(fmat[0]):
        try:
            desc = np.genfromtxt(fmat[0], dtype=float)
            print("loaded the descriptor matrix from file: ", fmat)
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    # sanity check
    if len(desc) == 0:
        raise ValueError('Please supply descriptor in a xyz file or a standlone descriptor matrix')
    print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))

    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")[:]
        ndict = len(tags)
    else:
        tags = []

    # scale & center
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print('Shape of descriptor matrix is {}'.format(desc.shape))
        print(scaler.fit(desc))
        desc = scaler.transform(desc)  # normalizing the features

    # fit UMAP

    reducer = umap.UMAP()
    proj = reducer.fit_transform(desc)
    if peratom or plotatomic and not projectatomic:
        proj_atomic_all = reducer.transform(desc_atomic)

    # save
    if output == 'matrix':
        np.savetxt(foutput + ".coord", proj, fmt='%4.8f', header='low D coordinates of samples')
        if peratom:  
            np.savetxt(foutput + "-atomic.coord", proj_atomic_all, fmt='%4.8f', header='low D coordinates of samples')
    if output == 'xyz':
        if os.path.isfile(foutput + ".xyz"):
            os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
        asapxyz.set_descriptors(proj, 'pca_coord')
        if peratom:
            asapxyz.set_atomic_descriptors(proj_atomic_all, 'pca_coord')
        # remove the raw descriptors
        if not keepraw:
            asapxyz.remove_descriptors(fmat)
            asapxyz.remove_atomic_descriptors(fmat)
        asapxyz.write(foutput)

    # color scheme
    plotcolor, plotcolor_peratom, colorlabel, colorscale = set_color_function(fcolor, asapxyz, colorscol, 0, (peratom or plotatomic), projectatomic)

    # make plot
    if plotatomic:
        outfile = 'UMAP_4_' + prefix + '-c-' + fcolor + '-plotatomic.png'
    else:
        outfile = 'UMAP_4_' + prefix + '-c-' + fcolor + '.png'

    fig_spec_dict = {
        'outfile': outfile,
        'show': False,
        'title': None,
        'xlabel': 'Principal Axis 1',
        'ylabel': 'Principal Axis 2',
        'xaxis': True,  'yaxis': True,
        'remove_tick': False,
        'rasterized': True,
        'fontsize': 16,
        'components':{ 
            "first_p": {"type": 'scatter', 'clabel': colorlabel},
            "second_p": {"type": 'annotate', 'adtext': adtext}
             }
        }
    asap_plot = Plotters(fig_spec_dict)
    asap_plot.plot(proj[::-1, [pc1, pc2]], plotcolor[::-1], [], tags)
    if peratom or plotatomic and not projectatomic:
        asap_plot.plot(proj_atomic_all[::-1, [pc1, pc2]], plotcolor_peratom[::-1],[],[])
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', nargs='+', type=str, required=True,
                        help='Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-tags', type=str, default='none',
                        help='Location of tags for the first M samples. Plot the tags on the PCA map.')
    parser.add_argument('-colors', type=str, default='none',
                        help='Location of a file or name of the tags in ase xyz file. It should contain properties for all samples (N floats) used to color the scatter plot')
    parser.add_argument('--colorscolumn', type=int, default=0,
                        help='The column number of the properties used for the coloring. Starts from 0.')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--output', type=str, default='matrix', help='The format for output files ([xyz], [matrix])')
    parser.add_argument('--peratom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom pca coordinates (True/False)?')
    parser.add_argument('--keepraw', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to keep the high dimensional descriptor when output xyz file (True/False)?')
    parser.add_argument('--scale', type=str2bool, nargs='?', const=True, default=False,
                        help='Scale the coordinates (True/False). Scaling highly recommended.')
    parser.add_argument('--d', type=int, default=2, help='number of embedded dimensions to keep')
    parser.add_argument('--dim1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--dim2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--projectatomic', type=str2bool, nargs='?', const=True, default=False,
                        help='Building the UMAP projection based on atomic descriptors instead of global ones (True/False)')
    parser.add_argument('--plotatomic', type=str2bool, nargs='?', const=True, default=False,
                        help='Plot the manifold coordinates of all atomic environments (True/False)')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to adjust the texts (True/False)?')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.tags, args.colors, args.colorscolumn, args.prefix, args.output, args.peratom,
         args.keepraw, args.scale, args.d, args.dim1, args.dim2, args.projectatomic, args.plotatomic, args.adjusttext)
