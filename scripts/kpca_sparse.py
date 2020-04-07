#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse

from asaplib.data import ASAPXYZ
from asaplib.io import str2bool
from asaplib.compressor import fps
from asaplib.pca import KernelPCA
from asaplib.plot import *


def main(fmat, fxyz, ftags, fcolor, colorscol, prefix, output, peratom, keepraw, n_sparse, kpca_d, pc1, pc2, projectatomic, plotatomic,
         adtext):
    """

    Parameters
    ----------
    fmat: Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.
    fxyz: Location of xyz file for reading the properties.
    ftags: Location of tags for the first M samples. Plot the tags on the (k)PCA map.
    fcolor: Location of a file or name of the tags in ase xyz file. It should contain properties for all samples (N floats) used to color the scatterplot'
    colorscol: The column number of the properties used for the coloring. Starts from 0.
    prefix: Filename prefix, default is ASAP
    output: The format for output files ([xyz], [matrix]). Default is xyz.
    peratom: Whether to output per atom pca coordinates (True/False)
    keepraw: Whether to keep the high dimensional descriptor when output is an xyz file (True/False)
    n_sparse: number of representative samples, default is 5% of the data
    kpca_d: Number of the principle components to keep
    pc1: Plot the projection along which principle axes
    pc2: Plot the projection along which principle axes
    plotatomic: Plot the PCA coordinates of all atomic environments (True/False)
    adtext: Whether to adjust the texts (True/False)

    Returns
    -------

    """
    use_atomic_desc = (peratom or plotatomic or projectatomic)
    # try to read the xyz file
    if fxyz != 'none':
        asapxyz = ASAPXYZ(fxyz)
        desc, desc_atomic = asapxyz.get_descriptors(fmat, use_atomic_desc)
    else:
        print("Did not provide the xyz file. We can only output descriptor matrix.")
        output = 'matrix'
    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(fmat):
        try:
            desc = np.genfromtxt(fmat, dtype=float)
            print("loaded the descriptor matrix from file: ", fmat)
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    # sanity check
    if len(desc) == 0:
        raise ValueError('Please supply descriptor in a xyz file or a standlone descriptor matrix')

    if projectatomic: desc = desc_atomic.copy()
    print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))

    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")[:]
        ndict = len(tags)

    # sparsification
    n_sample = len(desc)
    # set default value of n_sparse
    if n_sparse == 0:
        n_sparse = n_sample // 20
    # sparsification
    if n_sparse >= n_sample:
        print("the number of representative structure is too large, please select n < ", n_sample)
    elif n_sample > 0:
        ifps, _ = fps(desc, n_sparse, 0)
    else:
        print("Not using any sparsification")
        ifps = np.range(n_sample)

    # use linear kernel for now
    kNN = np.dot(desc[ifps],desc[ifps].T)
    kNM = np.dot(desc[:],desc[ifps].T)
    print("Shape of the kNN matrix: ", np.shape(kNN), ", and shape of the kNM matrix:", np.shape(kNM))
    # main thing
    kpca = KernelPCA(kpca_d)
    kpca.fit(kNN)
    proj = kpca.transform(kNM)

    # save
    if output == 'matrix':
        np.savetxt(prefix + "-kpca-d" + str(kpca_d) + ".coord", proj, fmt='%4.8f',
                   header='low D coordinates of samples')
    elif output == 'xyz':
        if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
        asapxyz.set_descriptors(proj, 'kpca_coord')
        asapxyz.write(foutput)

    # color scheme
    if plotatomic or projectatomic:
        plotcolor, plotcolor_peratom, colorlabel, colorscale = set_color_function(fcolor, fxyz, colorscol, asapxyz.get_num_frames(), True)
    else:
        plotcolor, colorlabel, colorscale = set_color_function(fcolor, fxyz, colorscol, len(proj))
    if projectatomic: plotcolor = plotcolor_peratom

    # make plot
    plot_styles.set_nice_font()
    # fig, ax = plt.subplots()

    fig, ax = plot_styles.plot_density_map(proj[:, [pc1, pc2]], plotcolor,
                                           xlabel='Principal Axis ' + str(pc1), ylabel='Principal Axis ' + str(pc2),
                                           clabel=colorlabel, label=None,
                                           xaxis=True, yaxis=True,
                                           centers=None,
                                           psize=30,
                                           out_file=None,
                                           title='KPCA for: ' + prefix,
                                           show=False, cmap='gnuplot',
                                           remove_tick=False,
                                           use_perc=True,
                                           rasterized=True,
                                           fontsize=15,
                                           vmax=colorscale[1],
                                           vmin=colorscale[0])

    fig.set_size_inches(18.5, 10.5)

    if ftags != 'none':
        texts = []
        for i in range(ndict):
            if tags[i] != 'None' and tags[i] != 'none' and tags[i] != '':
                ax.scatter(proj[i, pc1], proj[i, pc2], marker='^', c='black')
                texts.append(ax.text(proj[i, pc1], proj[i, pc2], tags[i],
                                     ha='center', va='center', fontsize=15, color='red'))
            # ax.annotate(tags[i], (proj[i,pc1], proj[i,pc2]))
        if (adtext):
            from adjustText import adjust_text
            adjust_text(texts, on_basemap=True,  # only_move={'points':'', 'text':'x'},
                        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                        force_text=(0.03, 0.5), force_points=(0.01, 0.25),
                        ax=ax, precision=0.01,
                        arrowprops=dict(arrowstyle="-", color='black', lw=1, alpha=0.8))

    plt.show()
    fig.savefig('KPCA_4_' + prefix + '-c-' + fcolor + '.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, required=True,
                        help='Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-tags', type=str, default='none',
                        help='Location of tags for the first M samples. Plot the tags on the KPCA map.')
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
    parser.add_argument('--n', type=int, default=0,
                        help='number of the representative samples, set negative if using no sparsification')
    parser.add_argument('--d', type=int, default=10, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--projectatomic', type=str2bool, nargs='?', const=True, default=False,
                        help='Building the KPCA projection based on atomic descriptors instead of global ones (True/False)')
    parser.add_argument('--plotatomic', type=str2bool, nargs='?', const=True, default=False,
                        help='Plot the KPCA coordinates of all atomic environments (True/False)')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to adjust the texts (True/False)?')

    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.tags, args.colors, args.colorscolumn, args.prefix, args.output, args.peratom,
         args.keepraw, args.n, args.d, args.pc1, args.pc2, args.projectatomic, args.plotatomic, args.adjusttext)
