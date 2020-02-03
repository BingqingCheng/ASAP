#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from asaplib.pca import kpca
from asaplib.kde import KDE
from asaplib.kernel import kerneltodis
from asaplib.cluster import get_cluster_size, get_cluster_properties
from asaplib.cluster import DBCluster, sklearn_DB, LAIO_DB
from asaplib.plot import plot_styles
from asaplib.io import str2bool


def main(fmat, kmat, ftags, prefix, fcolor, dimension, pc1, pc2, algorithm, adtext):

    if fmat == 'none' and kmat == 'none':
        raise ValueError('Must provide either the low-dimensional coordinates fmat or the kernel matrix kmat')

    if fmat != 'none':
        try:
            proj = np.genfromtxt(fmat, dtype=float)[:, 0:dimension]
        except:
            raise ValueError('Cannot load the coordinates')
        print("loaded coordinates ", fmat, "with shape", np.shape(proj))

    if kmat != 'none':
        try:
            kNN = np.genfromtxt(kmat, dtype=float)
        except:
            raise ValueError('Cannot load the coordinates')
        print("loaded kernal matrix", kmat, "with shape", np.shape(kmat))

    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    # do a low dimensional projection of the kernel matrix
    if kmat != 'none':
        proj = kpca(kNN, dimension)

    density_model = KDE()        
    # fit density model to data
    try:
        density_model.fit(proj)
    except:
        raise RuntimeError('KDE did not work. Try smaller dimension.')
    # the characteristic bandwidth of the data
    sigma_kij = density_model.bandwidth
    rho = density_model.evaluate_density(proj)
    meanrho = np.mean(rho)

    algorithm = str(algorithm)
    # now we do the clustering
    if algorithm == 'dbscan' or algorithm == 'DBSCAN':
        ''' option 1: do on the projected coordinates'''
        trainer = sklearn_DB(sigma_kij, 5, 'euclidean')  # adjust the parameters here!
        do_clustering = DBCluster(trainer) 
        do_clustering.fit(proj)

        ''' option 2: do directly on kernel matrix.'''
        #dmat = kerneltodis(kNN)
        #trainer = sklearn_DB(sigma_kij, 5, 'precomputed') # adjust the parameters here!
        #do_clustering = DBCluster(trainer) 
        #do_clustering.fit(dmat)

    elif algorithm == 'fdb' or algorithm == 'FDB':
        if kmat == 'none':
            kNN = np.dot(proj, proj.T)
            print("convert coordinates to kernal matrix with dimension: ", np.shape(kNN))
        dmat = kerneltodis(kNN)
        #trainer = LAIO_DB(-1, -1)  # adjust the parameters here!
        trainer = LAIO_DB()
        do_clustering = DBCluster(trainer) 
        #do_clustering.fit(dmat, rho)
        do_clustering.fit(proj)
    else:
        raise ValueError('Please select from fdb or dbscan')

    print(do_clustering.pack())
    labels_db = do_clustering.get_cluster_labels()
    n_clusters = do_clustering.get_n_cluster()

    # save
    np.savetxt(prefix+"-cluster-label.dat", np.transpose([np.arange(len(labels_db)), labels_db]), header='index cluster_label', fmt='%d %d')
    # properties of each cluster
    #[ unique_labels, cluster_size ]  = get_cluster_size(labels_db[:])
    # center of each cluster
    #[ unique_labels, cluster_x ]  = get_cluster_properties(labels_db[:],proj[:,pc1],'mean')
    #[ unique_labels, cluster_y ]  = get_cluster_properties(labels_db[:],proj[:,pc2],'mean')

    # color scheme
    fcolor = str(fcolor)
    if fcolor == 'rho': # we use the local density as the color scheme
        plotcolor = rho
        colorlabel = 'local density of each data point (bandwith $\sigma(k_{ij})$ ='+"{:4.0e}".format(sigma_kij)+' )'
    elif fcolor == 'cluster':
        plotcolor = labels_db
        colorlabel = 'a total of' + str(n_clusters) + ' clusters'
    else:
        try:
            plotcolor = np.genfromtxt(fcolor, dtype=float)
        except: raise ValueError('Cannot load the vector of properties')
        if len(plotcolor) != len(kNN):
            raise ValueError('Length of the vector of properties is not the same as number of samples')
        colorlabel = 'use '+fcolor+' for coloring the data points'
    [plotcolormin, plotcolormax] = [np.min(plotcolor), np.max(plotcolor)]

    # make plot
    plot_styles.set_nice_font()

    fig, ax = plot_styles.plot_cluster_w_size(proj[:, [pc1, pc2]], labels_db, rho, s=None,
                      clabel=colorlabel, title=None, 
                      w_size=True, w_label=True,
                      circle_size=20, alpha=0.5, edgecolors=None,
                      cmap='gnuplot', vmax=None,vmin=None, psize=20,
                      show=False, savefile=None, fontsize =15,
                      figsize=None,rasterized=True, remove_tick=True,
                      dpi=200, outlier=True)
    """
    ax = plot_styles.plot_cluster_w_label(proj[:,[pc1,pc2]], labels_db, Xcluster=None, 
                      show=False, savefile = None, fontsize =15, psize = 20, 
                      title=None, w_label = True, figsize=None,
                      dpi=200, alpha=0.7, edgecolors=None, cp_style=1, w_legend=False, outlier=True)
    """
    fig.set_size_inches(18.5, 10.5)

    # project the known structures
    if ftags != 'none':
        texts = []
        for i in range(ndict):
            ax.scatter(proj[i, pc1],proj[i, pc2],marker='^', c='black')
            texts.append(ax.text(proj[i, pc1],proj[i, pc2], tags[i],
                         ha='center', va='center', fontsize=15, color='red'))
            #ax.annotate(tags[i], (proj[i,pc1], proj[i,pc2]))
        if adtext:
            from adjustText import adjust_text
            adjust_text(texts,on_basemap=True,  # only_move={'points':'', 'text':'x'},
                    expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                   force_text=(0.03, 0.5), force_points=(0.01, 0.25),
                   ax=ax, precision=0.01,
                  arrowprops=dict(arrowstyle="-", color='black', lw=1, alpha=0.8))

    plt.title('PCA and clustering for: '+prefix)
    plt.xlabel('Princple Axis '+str(pc1))
    plt.ylabel('Princple Axis '+str(pc2))
    plt.show()
    fig.savefig('Clustering_4_'+prefix+'.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, default='none', help='Location of the low D projection of the data.')
    parser.add_argument('-kmat', type=str, default='none', help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('-colors', type=str, default='cluster', help='Properties for all samples (N floats) used to color the scatter plot,[filename/rho/cluster]')
    parser.add_argument('--d', type=int, default=8, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--algo', type=str, default='fdb', help='the algorithm for density-based clustering ([dbscan], [fdb])')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False, help='Do you want to adjust the texts (True/False)?')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fmat, args.kmat, args.tags, args.prefix, args.colors, args.d, args.pc1, args.pc2, args.algo, args.adjusttext)
