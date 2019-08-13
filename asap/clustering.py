#!/usr/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from lib import kpca, kerneltodis, KDE
from lib import get_cluster_size, get_cluster_properties
from lib import DBCluster, sklearn_DB, LAIO_DB
from lib import plot_styles

def main(fkmat, ftags, prefix, kpca_d, pc1, pc2, algorithm):

    # if it has been computed before we can simply load it
    try:
        eva = np.genfromtxt(fkmat, dtype=float)
    except: raise ValueError('Cannot load the kernel matrix')

    print("loaded",fkmat)
    if (ftags != 'none'): 
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    # do a low dimensional projection of the kernel matrix
    proj = kpca(eva,kpca_d)

    density_model = KDE()        
    # fit density model to data
    density_model.fit(proj)
    # the charecteristic bandwidth of the data        
    sigma_kij = density_model.bandwidth
    rho = density_model.evaluate_density(proj)
    meanrho = np.mean(rho)

    # now we do the clustering
    if (algorithm == 'dbscan' or algorithm == 'DBSCAN'):
        ''' option 1: do on the projected coordinates'''
        #trainer = sklearn_DB(sigma_kij, 5, 'euclidean') # adjust the parameters here!
        #do_clustering = DBCluster(trainer) 
        #do_clustering.fit(proj)

        ''' option 2: do directly on kernel matrix.'''
        dmat = kerneltodis(eva)
        trainer = sklearn_DB(sigma_kij, 5, 'precomputed') # adjust the parameters here!
        do_clustering = DBCluster(trainer) 
        do_clustering.fit(dmat)

    elif (algorithm == 'fdb' or algorithm == 'FDB'):
        dmat = kerneltodis(eva)
        trainer = LAIO_DB(-1,-1)
        do_clustering = DBCluster(trainer) # adjust the parameters here!
        do_clustering.fit(dmat, rho)
    else: raise ValueError('Please select from fdb or dbscan')

    print(do_clustering.pack())
    labels_db = do_clustering.get_cluster_labels()
    n_clusters = do_clustering.get_n_cluster()

    # save
    np.savetxt(prefix+"-cluster-label.dat", labels_db, fmt='%d')

    [ unique_labels, cluster_size ]  = get_cluster_size(labels_db[:])
    # center of each cluster
    [ unique_labels, cluster_x ]  = get_cluster_properties(labels_db[:],proj[:,pc1],'mean')
    [ unique_labels, cluster_y ]  = get_cluster_properties(labels_db[:],proj[:,pc2],'mean')

    # color scheme
    plotcolor = labels_db
    [ plotcolormin, plotcolormax ] = [ 0, n_clusters ]
    colorlabel = 'a total of' + str(n_clusters) + ' clusters'

    # make plot
    plot_styles.set_nice_font()
    """
    pcaplot = ax.scatter(proj[:,pc1],proj[:,pc2],c=plotcolor[:],
                    cmap=cm.gnuplot,vmin=plotcolormin, vmax=plotcolormax)
    cbar = fig.colorbar(pcaplot, ax=ax)
    cbar.ax.set_ylabel(colorlabel)

    # plot the clusters with size propotional to population
    for k in unique_labels:
        if (k >=0):
            ax.plot(cluster_x[k],cluster_y[k], 'o', markerfacecolor='none',
                markeredgecolor='gray', markersize=10.0*(np.log(cluster_size[k])))

    """
    fig, ax = plt.subplots()
    ax = plot_styles.plot_cluster_w_label(proj[:,[pc1,pc2]], labels_db, Xcluster=None, 
                      show=False, savefile = None, fontsize =15, psize = 20, 
                      title=None, w_label = True, figsize=None,
                      dpi=200, alpha=0.7, edgecolors=None, cp_style=1, w_legend=False, outlier=True)

    # project the known structures
    if (ftags != 'none'):
        for i in range(ndict):
            ax.scatter(proj[i,pc1],proj[i,pc2],marker='^',c='black')
            ax.annotate(tags[i], (proj[i,pc1], proj[i,pc2]))

    plt.title('KPCA and clustering for: '+prefix)
    plt.xlabel('Princple Axis '+str(pc1))
    plt.ylabel('Princple Axis '+str(pc2))
    plt.show()
    fig.savefig('Clustering_4_'+prefix+'.png')

##########################################################################################
##########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-kmat', type=str, required=True, help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--d', type=int, default=8, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--algo', type=str, default='fdb', help='the algotithm for density-based clustering ([dbscan], [fdb])')
    args = parser.parse_args()

    main(args.kmat, args.tags, args.prefix, args.d, args.pc1, args.pc2, args.algo)


