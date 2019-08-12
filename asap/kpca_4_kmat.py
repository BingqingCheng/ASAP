#!/usr/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from lib import kpca, kerneltorho

def main(fkmat, ftags, prefix, kpca_d, pc1, pc2):

    # if it has been computed before we can simply load it
    eva = np.genfromtxt(fkmat, dtype=float)
    print("loaded",fkmat)
    if (ftags != 'none'): 
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    # charecteristic difference in k_ij
    delta = np.std(eva[:,:])
    # Get local density
    rho = kerneltorho(eva, delta)
    [ rhomin, rhomax ] = [ np.min(rho),np.max(rho) ]

    # main thing
    proj = kpca(eva,kpca_d)

    # save
    np.savetxt(prefix+"-kpca-d"+str(kpca_d)+".coord", proj, fmt='%4.8f')

    # make plot
    fig, ax = plt.subplots()
    pcaplot = ax.scatter(proj[:,pc1],proj[:,pc2],c=rho[:],cmap=cm.cool,vmin=rhomin, vmax=rhomax)
    cbar = fig.colorbar(pcaplot, ax=ax)
    cbar.ax.set_ylabel('local density of each data point (delta ='+"{:4.0e}".format(delta)+' )')

    # project the known structures
    if (ftags != 'none'):
        for i in range(ndict):
            ax.scatter(proj[i,pc1],proj[i,pc2],marker='^',c='black')
            ax.annotate(tags[i], (proj[i,pc1], proj[i,pc2]))

    plt.title('KPCA for: '+prefix)
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    fig.set_size_inches(18.5, 10.5)
    plt.show()
    fig.savefig('KPCA_4_'+prefix+'.png')
##########################################################################################
##########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-kmat', type=str, default='none', help='Location of kernel matrix file')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for each sample')
    parser.add_argument('--prefix', type=str, default='', help='Filename prefix')
    parser.add_argument('--d', type=int, default=10, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    args = parser.parse_args()

    main(args.kmat, args.tags, args.prefix, args.d, args.pc1, args.pc2)


