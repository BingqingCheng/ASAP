#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse

from ase.io import write

from asaplib.io import str2bool
from asaplib.pca import KernelPCA
from asaplib.plot import *


def main(fmat, fxyz, ftags, fcolor, colorscol, prefix, output, kpca_d, pc1, pc2, adtext):
    # if it has been computed before we can simply load it
    try:
        kNN = np.genfromtxt(fmat, dtype=float)
    except:
        raise ValueError('Cannot load the kernel matrix')

    print("loaded", fmat)
    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")[:, 0]
        ndict = len(tags)

    # try to read the xyz file
    if fxyz != 'none':
        try:
            frames = read(fxyz, ':')
            nframes = len(frames)
            print('load xyz file: ', fxyz, ', a total of ', str(nframes), 'frames')
        except:
            raise ValueError('Cannot load the xyz file')

    if output == 'xyz' and fxyz == 'none':
        raise ValueError('Need input xyz in order to output xyz')

    # main thing
    proj = KernelPCA(kpca_d).fit_transform(kNN)

    # save
    if output == 'matrix':
        np.savetxt(prefix + "-kpca-d" + str(kpca_d) + ".coord", proj, fmt='%4.8f',
                   header='low D coordinates of samples')
    elif output == 'xyz':
        if len(frames) > 1:
            for i, frame in enumerate(frames):
                frame.info['kpca_coord'] = proj[i]
                write(prefix + "-kpca-d" + str(kpca_d) + ".xyz", frames[i], append=True)
        else:
            frames[0].new_array('kpca_coord', proj)
            write(prefix + "-kpca-d" + str(kpca_d) + ".xyz", frames[0], append=False)

    # color scheme
    plotcolor, colorlabel = set_color_function(fcolor, fxyz, colorscol, len(proj))

    # make plot
    plot_styles.set_nice_font()
    # fig, ax = plt.subplots()

    fig, ax = plot_styles.plot_density_map(proj[:, [pc1, pc2]], plotcolor,
                                           xlabel='Principal Axis ' + str(pc1), ylabel='Principal Axis ' + str(pc2),
                                           clabel=colorlabel, label=None,
                                           centers=None,
                                           psize=30,
                                           out_file='KPCA_4_' + prefix + '.png',
                                           title='KPCA for: ' + prefix,
                                           show=False, cmap='gnuplot',
                                           remove_tick=False,
                                           use_perc=True,
                                           rasterized=True,
                                           fontsize=15,
                                           vmax=None,
                                           vmin=None)

    fig.set_size_inches(18.5, 10.5)

    if ftags != 'none':
        texts = []
        for i in range(ndict):
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
                        help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('-colors', type=str, default='none',
                        help='Location of a file that contains properties for all samples (N floats) used to color the scatter plot')
    parser.add_argument('--colorscolumn', type=int, default=0,
                        help='The column number of the properties used for the coloring. Starts from 0.')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--output', type=str, default='matrix', help='The format for output files ([xyz], [matrix])')
    parser.add_argument('--d', type=int, default=10, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to adjust the texts (True/False)?')

    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.tags, args.colors, args.colorscolumn, args.prefix, args.output, args.d, args.pc1,
         args.pc2, args.adjusttext)
