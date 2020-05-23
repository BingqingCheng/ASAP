#!/usr/bin/python3
"""
script for making k-PCA map based on precomputed design matrix
"""

import argparse

from asaplib.data import ASAPXYZ
from asaplib.io import str2bool
from asaplib.pca import KernelPCA
from asaplib.plot import *


def main(fmat, fxyz, ftags, fcolor, colorscol, prefix, output, kpca_d, pc1, pc2, adtext, use_morgan, metric):
    """

    Parameters
    ----------
    fmat: Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.
    fxyz: Location of xyz file for reading the properties.
    ftags: Location of tags for the first M samples. Plot the tags on the umap.
    fcolor: Location of a file or name of the tags in ase xyz file. It should contain properties for all samples (N floats) used to color the scatterplot'
    colorscol: The column number of the properties used for the coloring. Starts from 0.
    prefix: Filename prefix, default is ASAP
    output: The format for output files ([xyz], [matrix]). Default is xyz.
    kpca_d: number of dimensions
    pc1: Plot the projection along which principle axes
    pc2: Plot the projection along which principle axes
    adtext: Whether to adjust the texts (True/False)
    use_morgan: Whether to use Morgan fingerprints. True for the photoswitch example.
    metric: KPCA metric to use if using Morgan fingerprints [linear, cosine, rbf, poly, sigmoid]

    Returns
    -------

    """

    if not use_morgan:

        foutput = prefix + "-kpca-d" + str(kpca_d)
        # load the kernel matrix
        try:
            kNN = np.genfromtxt(fmat, dtype=float)
        except:
            raise ValueError('Cannot load the kernel matrix')

        print("loaded", fmat)
        if ftags != 'none':
            tags = np.loadtxt(ftags, dtype="str")
            if tags.ndim > 1:
                tags = tags[:, 0]
            ndict = len(tags)

        asapxyz = None
        # try to read the xyz file
        if fxyz != 'none':
            asapxyz = ASAPXYZ(fxyz)
        elif output == 'xyz':
            print("Did not provide the xyz file. We can only output descriptor matrix.")
            output = 'matrix'

        # main thing
        proj = KernelPCA(kpca_d).fit_transform(kNN)

        # save
        if output == 'matrix':
            np.savetxt(prefix + "-kpca-d" + str(kpca_d) + ".coord", proj, fmt='%4.8f',
                       header='low D coordinates of samples')
        elif output == 'xyz':
            if os.path.isfile(foutput + ".xyz"):
                os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
            asapxyz.set_descriptors(proj, 'kpca_coord')
            asapxyz.write(foutput)

    else:

        if fxyz != 'none':
            asapxyz = ASAPXYZ(fxyz)
        else:
            raise Exception('xyz file required for plotting')

        # Use Morgan fingerprints

        import pandas as pd
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
        from sklearn.decomposition import kernel_pca

        path = 'photoswitches.csv'

        df = pd.read_csv(path)
        smiles_list = df['SMILES'].to_list()

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in rdkit_mols]
        X = np.asarray(X)

        reducer = kernel_pca.KernelPCA(kernel=metric, n_components=2)
        proj = reducer.fit_transform(X, )

    # color scheme
    plotcolor, colorlabel, colorscale = set_color_function(fcolor, asapxyz, colorscol, len(proj))

    # make plot
    plot_styles.set_nice_font()
    # fig, ax = plt.subplots()

    fig, ax = plot_styles.plot_density_map(proj[:, [pc1, pc2]], plotcolor,
                                           xlabel='Principal Axis ' + str(pc1), ylabel='Principal Axis ' + str(pc2),
                                           clabel=colorlabel, label=None,
                                           xaxis=True, yaxis=True,
                                           centers=None,
                                           psize=20,
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
        if adtext:
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
    parser.add_argument('-fmat', nargs='+', type=str, required=True,
                        help='Location of descriptor matrix file or name of the tags in ase xyz file. You can use gen_descriptors.py to compute it.')
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
    parser.add_argument('--use_morgan', type=bool, default=False, help='Whether or not to use Morgan fingerprints')
    parser.add_argument('--metric', type=str, default='cosine', help='Which kernel metric to use if using Morgan'
                                                                     'fingerprints')
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.tags, args.colors, args.colorscolumn, args.prefix, args.output, args.d, args.pc1,
         args.pc2, args.adjusttext, args.use_morgan, args.metric)
