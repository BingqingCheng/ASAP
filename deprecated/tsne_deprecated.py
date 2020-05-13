#!/usr/bin/python3
"""
script for applying t-SNE to a precomputed design matrix. See: https://lvdmaaten.github.io/tsne/
for getting the most out of t-SNE for your example.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from asaplib.data import ASAPXYZ
from asaplib.io import str2bool
from asaplib.plot import Plotters, set_color_function


def main(fmat, fxyz, ftags, fcolor, colorscol, prefix, output, peratom, keepraw, scale, tsne_d, dim1, dim2, perplexity,
         projectatomic, plotatomic, adtext):
    """

    Parameters
    ----------
    fmat: Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.
    fxyz: Location of xyz file for reading the properties.
    ftags: Location of tags for the first M samples. Plot the tags on the t-SNE map.
    fcolor: Location of a file or name of the tags in ase xyz file. It should contain properties for all samples (N floats) used to color the scatterplot'
    colorscol: The column number of the properties used for the coloring. Starts from 0.
    prefix: Filename prefix, default is ASAP
    output: The format for output files ([xyz], [matrix]). Default is xyz.
    peratom: Whether to output per atom t-SNE coordinates (True/False)
    keepraw: Whether to keep the high dimensional descriptor when output is an xyz file (True/False)
    scale: Scale the coordinates (True/False). Scaling highly recommanded.
    tsne_d: Dimension of the embedded space.
    dim1: Plot the projection along which principle axes
    dim2: Plot the projection along which principle axes
    projectatomic: build the projection using the (big) atomic descriptor matrix
    perplexity: Perplexity setting for t-SNE: Typical values between 5 and 50.
    plotatomic: Plot the PCA coordinates of all atomic environments (True/False)
    adtext: Whether to adjust the texts (True/False)

    Returns
    -------

    """

    foutput = prefix + "-pca-d" + str(tsne_d)
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

    # fit t-SNE

    if desc.shape[1] >= 50:
        # pre-process with PCA if dim > 50
        # suggested here: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

        pca = PCA(n_components=50)
        desc = pca.fit_transform(desc)
        print('Shape of processed descriptor matrix after applying PCA is {}'.format(desc.shape))

    tsne = TSNE(n_components=tsne_d, perplexity=perplexity)
    proj = tsne.fit_transform(desc)
    if peratom or plotatomic and not projectatomic:
        raise NotImplementedError
        #proj_atomic_all = tsne.transform(desc_atomic)

    # save
    if output == 'matrix':
        np.savetxt(foutput + ".coord", proj, fmt='%4.8f', header='low D coordinates of samples')
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
    plot_styles.set_nice_font()
    fig, ax = plt.subplots()
    if plotatomic and not projectatomic:
        # notice that we reverse the list of coordinates, in order to make the structures in the dictionary more obvious
        fig, ax = plot_styles.plot_density_map(proj_atomic_all[::-1, [dim1, dim2]], plotcolor_peratom[::-1], fig, ax,
                                               xlabel='Principal Axis ' + str(dim1),
                                               ylabel='Principal Axis ' + str(dim2),
                                               clabel=None, label=None,
                                               xaxis=True, yaxis=True,
                                               centers=None,
                                               psize=None,
                                               out_file=None,
                                               title=None,
                                               show=False, cmap='gnuplot',
                                               remove_tick=False,
                                               use_perc=False,
                                               rasterized=True,
                                               fontsize=15,
                                               vmax=colorscale[1],
                                               vmin=colorscale[0])

    fig, ax = plot_styles.plot_density_map(proj[::-1, [dim1, dim2]], plotcolor[::-1], fig, ax,
                                           xlabel='Principal Axis ' + str(dim1), ylabel='Principal Axis ' + str(dim2),
                                           clabel=colorlabel, label=None,
                                           xaxis=True, yaxis=True,
                                           centers=None,
                                           psize=None,
                                           out_file=None,
                                           title='t-SNE for: ' + prefix,
                                           show=False, cmap='gnuplot',
                                           remove_tick=False,
                                           use_perc=False,
                                           rasterized=True,
                                           fontsize=15,
                                           vmax=colorscale[1],
                                           vmin=colorscale[0])

    fig.set_size_inches(160.5, 80.5)

    if ftags != 'none':
        texts = []
        for i in range(ndict):
            if tags[i] != 'None' and tags[i] != 'none' and tags[i] != '':
                ax.scatter(proj[i, dim1], proj[i, dim2], marker='^', c='black')
                texts.append(ax.text(proj[i, dim1], proj[i, dim2], tags[i],
                                     ha='center', va='center', fontsize=10, color='red'))
        if adtext:
            from adjustText import adjust_text
            adjust_text(texts, on_basemap=True,  # only_move={'points':'', 'text':'x'},
                        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                        force_text=(0.03, 0.5), force_points=(0.01, 0.25),
                        ax=ax, precision=0.01,
                        arrowprops=dict(arrowstyle="-", color='black', lw=1, alpha=0.8))

    plt.show()
    if plotatomic:
        fig.savefig('t-SNE_4_' + prefix + '-c-' + fcolor + '-plotatomic.png')
    else:
        fig.savefig('t-SNE_4_' + prefix + '-c-' + fcolor + '.png')


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
    parser.add_argument('--perp', type=int, default=30, help='Perplexity value for t-SNE. Typical values between 5 and 50.')
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
         args.keepraw, args.scale, args.d, args.dim1, args.dim2, args.perp, args.projectatomic, args.plotatomic, args.adjusttext)
