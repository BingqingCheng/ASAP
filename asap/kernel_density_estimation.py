#!/usr/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from lib import kpca, KDE
from lib import plot_styles

def main(fkmat, ftags, fcolor, prefix, kpca_d, pc1, pc2):

    # if it has been computed before we can simply load it
    try:
        eva = np.genfromtxt(fkmat, dtype=float)
    except: raise ValueError('Cannot load the kernel matrix')
    print("loaded",fkmat)
    # load tags if any
    if (ftags != 'none'): 
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    # doing a kpca
    proj = kpca(eva,kpca_d)
    # save the low D projection
    # np.savetxt(prefix+"-kpca-d"+str(kpca_d)+".coord", proj, fmt='%4.8f', header='low D coordinates of samples')

    density_model = KDE()        
    # fit density model to data
    density_model.fit(proj)        
    sigma_kij = density_model.bandwidth
    rho = density_model.evaluate_density(proj)
    # save the density
    np.savetxt(prefix+"-kde.dat",np.transpose([np.arange(len(rho)),rho]), header='index kernel_density_estimation',fmt='%d %4.8f')

    # color scheme
    if (fcolor != 'none'):
        try:
            plotcolor = np.genfromtxt(fcolor, dtype=float)
        except: raise ValueError('Cannot load the vector of properties')
        if (len(plotcolor) != len(eva)): 
            raise ValueError('Length of the vector of properties is not the same as number of samples')
        colorlabel = 'use '+fcolor+' for coloring the data points'
    else: # we use the local density as the color scheme
        plotcolor = rho
        colorlabel = 'local density of each data point (bandwith $\sigma(k_{ij})$ ='+"{:4.0e}".format(sigma_kij)+' )'
    [ plotcolormin, plotcolormax ] = [ np.min(plotcolor),np.max(plotcolor) ]

    # make plot
    plot_styles.set_nice_font()
    fig, ax = plt.subplots()
    # density plot
    plot_styles.plot_density_map(proj[:,[pc1,pc2]], plotcolor,
                xlabel='Princple Axis '+str(pc1), ylabel='Princple Axis '+str(pc2), 
                clabel=colorlabel, label=None,
                centers=None,
                psize=20,
                out_file='KPCA_4_'+prefix+'.png', 
                title='KPCA for: '+prefix, 
                show=False, cmap='summer',
                remove_tick=False,
                use_perc=False,
                rasterized = True,
                fontsize = 15,
                vmax = plotcolormax,
                vmin = plotcolormin)

    # project the known structures
    if (ftags != 'none'):
        for i in range(ndict):
            plt.scatter(proj[i,pc1],proj[i,pc2],marker='^',c='black')
            plt.annotate(tags[i], (proj[i,pc1], proj[i,pc2]))

    plt.show()
    fig.savefig('kde_4_'+prefix+'.png')

##########################################################################################
##########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-kmat', type=str, required=True, help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('-colors', type=str, default='none', help='Properties for all samples (N floats) used to color the scatter plot')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--d', type=int, default=10, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    args = parser.parse_args()

    main(args.kmat, args.tags, args.colors, args.prefix, args.d, args.pc1, args.pc2)


