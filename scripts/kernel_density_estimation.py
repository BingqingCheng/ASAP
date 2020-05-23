#!/usr/bin/python3
"""
Script for doing kernel density estimation + plotting
from a low dimensional projection of the data set
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from asaplib.data import ASAPXYZ
from asaplib.io import str2bool
from asaplib.kde import KDE_internal, KDE_scipy, KDE_sklearn
from asaplib.plot import plot_styles


def main(fmat, fxyz, ftags, prefix, dimension, pc1, pc2, adtext):
    """

    Parameters
    ----------
    fmat: Location of low-dimensional coordinate file.
    ftags: Location of tags for the first M samples.
    prefix: Filename prefix.
    dimension: Number of the first X dimensions to keep
    pc1: First principle axis (int)
    pc2: Second principle axis (int)
    adtext: Boolean giving whether to adjust text or not.

    Returns
    -------

    """

    # try to read the xyz file
    if fxyz != 'none':
        asapxyz = ASAPXYZ(fxyz)
        desc, _ = asapxyz.get_descriptors(fmat)
    if os.path.isfile(fmat):
        try:
            desc = np.genfromtxt(fmat, dtype=float)
            print("loaded the descriptor matrix from file: ", fmat)
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    if len(desc) == 0:
        raise ValueError('Please supply descriptor in a xyz file or a standlone descriptor matrix')
    print("loaded", fmat, " with shape", np.shape(desc))
    # load tags if any
    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    proj = np.asmatrix(desc)[:, 0:dimension]
    density_model = KDE_internal() # KDE_sklearn(bandwidth=1) # KDE_scipy()
    # fit density model to data
    try:
        density_model.fit(proj)
    except:
        raise RuntimeError('KDE did not work. Try smaller dimension.')

    rho = density_model.evaluate_density(proj)
    # save the density
    np.savetxt(prefix + "-kde.dat", np.transpose([np.arange(len(rho)), rho]),
               header='index log_of_kernel_density_estimation', fmt='%d %4.8f')

    # color scheme
    plotcolor = rho
    colorlabel = 'Log of density for every point'
    [plotcolormin, plotcolormax] = [np.min(plotcolor), np.max(plotcolor)]

    # make plot
    plot_styles.set_nice_font()
    # density plot
    fig, ax = plot_styles.plot_density_map(np.asarray(proj[:, [pc1, pc2]]), plotcolor,
                                           xlabel='Princple Axis ' + str(pc1), ylabel='Princple Axis ' + str(pc2),
                                           clabel=colorlabel, label=None,
                                           xaxis=True, yaxis=True,
                                           centers=None,
                                           psize=None,
                                           out_file=None,
                                           title='KDE for: ' + prefix,
                                           show=False, cmap='gnuplot',
                                           remove_tick=False,
                                           use_perc=False,
                                           rasterized=True,
                                           fontsize=15,
                                           vmax=plotcolormax,
                                           vmin=plotcolormin)

    fig.set_size_inches(18.5, 10.5)
    if ftags != 'none':
        texts = []
        for i in range(ndict):
            if tags[i] != 'None' and tags[i] != 'none' and tags[i] != '':
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

    plt.show()
    fig.savefig('kde_4_' + prefix + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, required=True,
                        help='Location of low dimensional coordinate file or name of the tags in ase xyz file.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--d', type=int, default=10, help='number of the first X dimensions to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to adjust the texts (True/False)?')
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.tags, args.prefix, args.d, args.pc1, args.pc2, args.adjusttext)
