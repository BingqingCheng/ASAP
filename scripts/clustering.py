#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from asaplib.data import ASAPXYZ
from asaplib.pca import PCA, KernelPCA
from asaplib.kernel import kerneltodis
from asaplib.cluster import DBCluster, sklearn_DB, LAIO_DB
from asaplib.plot import *
from asaplib.io import str2bool


def main(fmat, kmat, fxyz, ftags, prefix, fcolor, colorscol, dimension, pc1, pc2, algorithm, projectatomic, adtext):

    """

    Parameters
    ----------
    fmat: Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.
    kmat: Location of the kernel matrix.
    fxyz: Location of xyz file for reading the properties.
    ftags: Location of tags for the first M samples
    prefix: Filename prefix. Default is ASAP.
    fcolor: Properties for all samples (N floats) used to color the scatter plot,[filename/rho/cluster]
    colorscol: The column number of the properties used for the coloring. Starts from 0.
    dimension: The number of principle components to keep
    pc1: int, default is 0, which principle axis to plot the projection on
    pc2: int, default is 1, which principle axis to plot the projection on
    algorithm: the algorithm for density-based clustering options are: ([dbscan], [fdb])
    projectatomic: build the projection using the (big) atomic descriptor matrix
    adtext: Whether to adjust the text (True/False)

    Returns
    -------
    cluster labels, PCA plots
    """

    if fmat == 'none' and kmat == 'none':
        raise ValueError('Must provide either the low-dimensional coordinates fmat or the kernel matrix kmat')

    # try to read the xyz file
    if fxyz != 'none':
        asapxyz = ASAPXYZ(fxyz)
        desc, desc_atomic = asapxyz.get_descriptors(fmat, projectatomic)
        if projectatomic:
            desc = desc_atomic.copy()
    else:
        asapxyz = None
        print("Did not provide the xyz file. We can only output descriptor matrix.")
        output = 'matrix'

    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(fmat):
        try:
            desc = np.genfromtxt(fmat, dtype=float)
            print("loaded the descriptor matrix from file: ", fmat)
        except:
            raise ValueError('Cannot load the descriptor matrix from file')

    if kmat != 'none':
        try:
            kNN = np.genfromtxt(kmat, dtype=float)
            print("loaded kernal matrix", kmat, "with shape", np.shape(kmat))
            desc =  kerneltodis(kNN)
        except:
            raise ValueError('Cannot load the coordinates')
        

    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    # now we do the clustering
    if algorithm == 'dbscan':
        # we compute the characteristic bandwidth of the data
        # first select a subset of structures (20)
        sbs = np.random.choice(np.asarray(range(len(desc))), 50, replace=False)
        # the characteristic bandwidth of the data
        sigma_kij = np.percentile(cdist(desc[sbs], desc, 'euclidean'), 100*10./len(desc))
        ''' option 1: do on the projected coordinates'''
        trainer = sklearn_DB(sigma_kij, 5, 'euclidean')  # adjust the parameters here!
        do_clustering = DBCluster(trainer)
        do_clustering.fit(desc)

        ''' option 2: do directly on kernel matrix.'''
        # trainer = sklearn_DB(sigma_kij, 5, 'precomputed') # adjust the parameters here!
        # do_clustering = DBCluster(trainer)
        # do_clustering.fit(desc)

    elif algorithm == 'fdb' or algorithm == 'FDB':
        trainer = LAIO_DB()
        do_clustering = DBCluster(trainer)
        do_clustering.fit(desc)
        #rho_min, delta_min = do_clustering.pack()
        #fdb_param_dict = {'rho_min': rho_min, 'delta_min': delta_min}
        #np.savetxt(prefix + "FDB parameters", fdb_param_dict, header='FDB parameters', fmt='%d $d')
    else:
        raise ValueError('Please select from fdb or dbscan')

    print(do_clustering.pack())
    labels_db = do_clustering.get_cluster_labels()
    n_clusters = do_clustering.get_n_cluster()
    
    if asapxyz is not None and projectatomic:
        asapxyz.set_atomic_descriptors(labels_db, 'cluster_label')
    elif asapxyz is not None:
        asapxyz.set_descriptors(labels_db, 'cluster_label')

    # save
    np.savetxt(prefix + "-cluster-label.dat", np.transpose([np.arange(len(labels_db)), labels_db]),
               header='index cluster_label', fmt='%d %d')

    if  fmat != 'none':
        pca = PCA(dimension, True)
        proj = pca.fit_transform(desc)
    elif  kmat != 'none':
        proj = KernelPCA(dimension).fit_transform(kNN)

    # color scheme
    if fcolor == 'cluster_label': 
        plotcolor = labels_db
        colorlabel = 'cluster_label'
    else:
        if projectatomic:
            _, plotcolor, colorlabel, colorscale = set_color_function(fcolor, asapxyz, colorscol, 0, True)
        else:
            plotcolor, colorlabel, colorscale = set_color_function(fcolor, asapxyz, colorscol, len(proj), False)

    # make plot
    plot_styles.set_nice_font()

    fig, ax = plot_styles.plot_cluster_w_size(proj[:, [pc1, pc2]], labels_db, plotcolor, s=None,
                                              clabel=colorlabel, title=None,
                                              w_size=True, w_label=True,
                                              circle_size=30, alpha=0.5, edgecolors=None,
                                              cmap='gnuplot', vmax=None, vmin=None, psize=20,
                                              show=False, savefile=None, fontsize=25,
                                              figsize=None, rasterized=True, remove_tick=True,
                                              dpi=200, outlier=True)
    """
    ax = plot_styles.plot_cluster_w_label(proj[:,[pc1,pc2]], labels_db, Xcluster=None, 
                      show=False, savefile = None, fontsize =15, psize = 20, 
                      title=None, w_label = True, figsize=None,
                      dpi=200, alpha=0.7, edgecolors=None, cp_style=1, w_legend=False, outlier=True)
    """
    fig.set_size_inches(160.5, 80.5)

    # project the known structures
    if ftags != 'none':
        texts = []
        for i in range(ndict):
            ax.scatter(proj[i, pc1], proj[i, pc2], marker='^', c='black')
            texts.append(ax.text(proj[i, pc1], proj[i, pc2], tags[i],
                                 ha='center', va='center', fontsize=15, color='red'))
            # ax.annotate(tags[i], (proj[i,pc1], proj[i,pc2]))
        if adtext:
            from adjustText import adjust_text
            adjust_text(texts, on_basemap=True,  # only_move={'points':'', 'text':'x'},
                        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                        force_text=(0.03, 0.5), force_points=(0.01, 0.25),
                        ax=ax, precision=0.01,
                        arrowprops=dict(arrowstyle="-", color='black', lw=1, alpha=0.8))

    plt.title('PCA and clustering for: ' + prefix)
    plt.xlabel('Princple Axis ' + str(pc1))
    plt.ylabel('Princple Axis ' + str(pc2))
    plt.show()
    fig.savefig('Clustering_4_' + prefix + '.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, default='none',
                        help='Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.')
    parser.add_argument('-kmat', type=str, default='none',
                        help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('-colors', type=str, default='cluster_label',
                        help='Properties for all samples (N floats) used to color the scatter plot')
    parser.add_argument('--colorscolumn', type=int, default=0,
                        help='The column number of the properties used for the coloring. Starts from 0.')
    parser.add_argument('--d', type=int, default=8, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--algo', type=str, default='fdb',
                        help='the algorithm for density-based clustering ([dbscan], [fdb])')
    parser.add_argument('--projectatomic', type=str2bool, nargs='?', const=True, default=False,
                        help='Building the KPCA projection based on atomic descriptors instead of global ones (True/False)')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to adjust the texts (True/False)?')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fmat, args.kmat, args.fxyz, args.tags, args.prefix, args.colors, args.colorscolumn, args.d,
         args.pc1, args.pc2, args.algo, args.projectatomic, args.adjusttext)
