#!/usr/bin/python3
"""
TODO: Module-level description
"""

import argparse
import sys

from ase.io import write

from asaplib.io import str2bool
from asaplib.pca import pca, pca_project
from asaplib.plot import *


def main(fmat, fxyz, ftags, fcolor, colorscol, prefix, output, peratom, keepraw, scale, pca_d, pc1, pc2, plotatomic,
         adtext):
    """

    Parameters
    ----------
    fmat: Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.
    fxyz: Location of xyz file for reading the properties.
    ftags: Location of tags for the first M samples. Plot the tags on the PCA map.
    fcolor: Location of a file or name of the tags in ase xyz file. It should contain properties for all samples (N floats) used to color the scatterplot'
    colorscol: The column number of the properties used for the coloring. Starts from 0.
    prefix: Filename prefix, default is ASAP
    output: The format for output files ([xyz], [matrix]). Default is xyz.
    peratom: Whether to output per atom pca coordinates (True/False)
    keepraw: Whether to keep the high dimensional descriptor when output is an xyz file (True/False)
    scale: Scale the coordinates (True/False). Scaling highly recommanded.
    pca_d: Number of the principle components to keep
    pc1: Plot the projection along which principle axes
    pc2: Plot the projection along which principle axes
    plotatomic: Plot the PCA coordinates of all atomic environments (True/False)
    adtext: Whether to adjust the texts (True/False)

    Returns
    -------

    """

    foutput = prefix + "-pca-d" + str(pca_d)
    peratom = bool(peratom)
    keepraw = bool(keepraw)
    plotatomic = bool(plotatomic)
    adtext = bool(adtext)
    scale = bool(scale)
    total_natoms = 0

    if output == 'xyz' and fxyz == 'none':
        raise ValueError('Need input xyz in order to output xyz')

    desc = [];
    ndesc = 0
    # try to read the xyz file
    if fxyz != 'none':
        try:
            frames = read(fxyz, ':')
            nframes = len(frames)
            print('load xyz file: ', fxyz, ', a total of ', str(nframes), 'frames')
        except:
            raise ValueError('Cannot load the xyz file')

        # load from xyz file
        if nframes > 1:
            for i, frame in enumerate(frames):
                total_natoms += len(frame.get_positions())
                if fmat in frame.info:
                    try:
                        desc.append(frame.info[fmat])
                        if ndesc > 0 and len(frame.info[fmat]) != ndesc:
                            raise ValueError('mismatch of number of descriptors between frames')
                        ndesc = len(frame.info[fmat])
                    except:
                        raise ValueError('Cannot combine the descriptor matrix from the xyz file')
            if desc != [] and np.shape(desc)[1] != nframes:
                desc = np.asmatrix(desc)
                # print(np.shape(desc))
                desc.reshape((ndesc, nframes))
        else:
            # only one frame
            total_natoms = len(frames[0].get_positions())
            try:
                desc = frames[0].get_array(fmat)
            except:
                ValueError('Cannot read the descriptor matrix from single frame')
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
    if len(desc) == 0:
        raise ValueError('Please supply descriptor in a xyz file or a standlone descriptor matrix')
    print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))

    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")[:]
        ndict = len(tags)

    # scale & center
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print(scaler.fit(desc))
        desc = scaler.transform(desc)  # normalizing the features

    # main thing
    proj, pvec = pca(desc, pca_d)
    proj_atomic_all = np.zeros((total_natoms, pca_d), dtype=float)
    # print(total_natoms)

    # save
    if output == 'matrix':
        np.savetxt(foutput + ".coord", proj, fmt='%4.8f', header='low D coordinates of samples')
    if output == 'xyz' or peratom or plotatomic:
        if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
        if nframes > 1:
            atom_index = 0
            for i, frame in enumerate(frames):
                frame.info['pca_coord'] = proj[i]
                # !!! this is the bits for per_atom proj
                if peratom or plotatomic:
                    try:
                        desc_atomic = frame.get_array(fmat)
                    except:
                        ValueError('Cannot read the descriptor per atom for frame ' + str(i))
                    desc_atomic = scaler.transform(desc_atomic)  # normalizing the features
                    proj_atomic = pca_project(desc_atomic, pvec)
                    if peratom: frame.new_array('pca_coord', proj_atomic)
                    if plotatomic:
                        natomnow = len(frame.get_positions())
                        proj_atomic_all[atom_index:atom_index + natomnow, :] = proj_atomic[:, :]
                        atom_index += natomnow
                # remove the raw descriptors
                if not keepraw:
                    frame.info[fmat] = None
                    frame.set_array(fmat, None)

                if output == 'xyz': write(foutput + ".xyz", frame, append=True)
        else:
            frames[0].new_array('pca_coord', proj)
            if output == 'xyz': write(prefix + "-pca-d" + str(pca_d) + ".xyz", frames[0], append=False)

    # color scheme
    if plotatomic:
        plotcolor, plotcolor_peratom, colorlabel = set_color_function(fcolor, fxyz, colorscol, len(proj), True)
    else:
        plotcolor, colorlabel = set_color_function(fcolor, fxyz, colorscol, len(proj), False)

    # make plot
    plot_styles.set_nice_font()
    fig, ax = plt.subplots()
    if plotatomic:
        # notice that we reverse the list of coordinates, in order to make the structures in the dictionary more obvious
        fig, ax = plot_styles.plot_density_map(proj_atomic_all[::-1, [pc1, pc2]], plotcolor_peratom[::-1], fig, ax,
                                               xlabel='Principal Axis ' + str(pc1), ylabel='Principal Axis ' + str(pc2),
                                               clabel=None, label=None,
                                               xaxis=True, yaxis=True,
                                               centers=None,
                                               psize=20,
                                               out_file=None,
                                               title=None,
                                               show=False, cmap='gnuplot',
                                               remove_tick=False,
                                               use_perc=False,
                                               rasterized=True,
                                               fontsize=15,
                                               vmax=None,
                                               vmin=None)

    fig, ax = plot_styles.plot_density_map(proj[::-1, [pc1, pc2]], plotcolor[::-1], fig, ax,
                                           xlabel='Principal Axis ' + str(pc1), ylabel='Principal Axis ' + str(pc2),
                                           clabel=colorlabel, label=None,
                                           xaxis=True, yaxis=True,
                                           centers=None,
                                           psize=200,
                                           out_file='PCA_4_' + prefix + '.png',
                                           title='PCA for: ' + prefix,
                                           show=False, cmap='gnuplot',
                                           remove_tick=False,
                                           use_perc=False,
                                           rasterized=True,
                                           fontsize=15,
                                           vmax=None,
                                           vmin=None)

    fig.set_size_inches(160.5, 80.5)

    if ftags != 'none':
        texts = []
        for i in range(ndict):
            if tags[i] != 'None' and tags[i] != 'none' and tags[i] != '':
                ax.scatter(proj[i, pc1], proj[i, pc2], marker='^', c='black')
                texts.append(ax.text(proj[i, pc1], proj[i, pc2], tags[i],
                                 ha='center', va='center', fontsize=10, color='red'))
        if adtext:
            from adjustText import adjust_text
            adjust_text(texts, on_basemap=True,  # only_move={'points':'', 'text':'x'},
                        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                        force_text=(0.03, 0.5), force_points=(0.01, 0.25),
                        ax=ax, precision=0.01,
                        arrowprops=dict(arrowstyle="-", color='black', lw=1, alpha=0.8))

    plt.show()
    fig.savefig('PCA_4_' + prefix + '-c-' + fcolor + '.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', type=str, default='ASAP_desc',
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
    parser.add_argument('--scale', type=str2bool, nargs='?', const=True, default=True,
                        help='Scale the coordinates (True/False). Scaling highly recommanded.')
    parser.add_argument('--d', type=int, default=10, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--plotatomic', type=str2bool, nargs='?', const=True, default=False,
                        help='Plot the PCA coordinates of all atomic environments (True/False)')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to adjust the texts (True/False)?')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.tags, args.colors, args.colorscolumn, args.prefix, args.output, args.peratom,
         args.keepraw, args.scale, args.d, args.pc1, args.pc2, args.plotatomic, args.adjusttext)
