#!/usr/bin/python3
"""
Sparse KPCA with plotting
"""

import argparse

from asaplib.data import ASAPXYZ
from asaplib.io import str2bool
from asaplib.compressor import fps, CUR_deterministic
from asaplib.kernel import Descriptors_to_Kernels
from asaplib.pca import KernelPCA
from asaplib.plot import *

def main(fmat, fxyz, ftags, fcolor, colorscol, prefix, output, peratom, keepraw, sparse_mode, n_sparse, power, kpca_d, pc1, pc2, projectatomic, plotatomic, adjusttext):

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
    power: use polynomial kernel function of degree n. 
    kpca_d: Number of the principle components to keep
    pc1: Plot the projection along which principle axes
    pc2: Plot the projection along which principle axes
    projectatomic: build the projection using the (big) atomic descriptor matrix
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
        if projectatomic: desc = desc_atomic.copy()
    else:
        asapxyz = None
        print("Did not provide the xyz file. We can only output descriptor matrix.")
        output = 'matrix'
    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(fmat[0]):
        try:
            desc = np.genfromtxt(fmat[0], dtype=float)
            print("loaded the descriptor matrix from file: ", fmat)
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    # sanity check
    if len(desc) == 0:
        raise ValueError('Please supply descriptor in a xyz file or a standlone descriptor matrix')

    print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))

    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")[:]
        ndict = len(tags)
    else:
        tags = []

    # sparsification
    n_sample = len(desc)
    # set default value of n_sparse
    if n_sparse == 0:
        n_sparse = max(10, n_sample // 20)
    # sparsification
    if n_sparse >= n_sample:
        print("the number of representative structure is too large, please select n < ", n_sample)
    elif n_sample > 0:
        if sparse_mode == 'fps' or sparse_mode == 'FPS':
            ifps, _ = fps(desc, n_sparse, 0)
        elif sparse_mode == 'cur' or sparse_mode == 'CUR':
            cov = np.dot(np.asmatrix(desc), np.asmatrix(desc).T)
            ifps, _ = CUR_deterministic(cov, n_sparse)
        else:
            raise ValueError('Cannot find the specified sparsification mode')
    else:
        print("Not using any sparsification")
        ifps = np.range(n_sample)

    k_spec = {'k0':{"type": "cosine"}} #{ 'k1': {"type": "polynomial", "d": power}}
    k_transform = Descriptors_to_Kernels(k_spec)

    kNN = k_transform.compute(desc[ifps])
    kMN = k_transform.compute(desc, desc[ifps])
    print("Shape of the kNN matrix: ", np.shape(kNN), ", and shape of the kMN matrix:", np.shape(kMN))
    # main thing
    kpca = KernelPCA(kpca_d)
    kpca.fit(kNN)
    proj = kpca.transform(kMN)
    if peratom or plotatomic and not projectatomic:
        kNT = np.power(np.dot(desc_atomic[:],desc[ifps].T),power)
        proj_atomic_all = kpca.transform(kNT)

    # save
    if output == 'matrix':
        np.savetxt(prefix + "-kpca-d" + str(kpca_d) + ".coord", proj, fmt='%4.8f',
                   header='low D coordinates of samples')
    elif output == 'xyz':
        if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
        asapxyz.set_descriptors(proj, 'kpca_coord')
        asapxyz.write(foutput)

    # color scheme
    plotcolor, plotcolor_peratom, colorlabel, colorscale = set_color_function(fcolor, asapxyz, colorscol, 0, (peratom or plotatomic), projectatomic)

    # make plot
    if plotatomic:
        outfile = 'KPCA_4_' + prefix + '-c-' + fcolor + '-plotatomic.png'
    else:
        outfile = 'KPCA_4_' + prefix + '-c-' + fcolor + '.png'

    fig_spec_dict = {
        'outfile': outfile,
        'show': False,
        'title': None,
        'xlabel': 'Principal Axis 1',
        'ylabel': 'Principal Axis 2',
        'xaxis': True,  'yaxis': True,
        'remove_tick': False,
        'rasterized': True,
        'fontsize': 16,
        'components':{ 
            "first_p": {"type": 'scatter', 'clabel': colorlabel},
            "second_p": {"type": 'annotate', 'adtext': adjusttext}
             }
        }
    asap_plot = Plotters(fig_spec_dict)
    asap_plot.plot(proj[::-1, [pc1, pc2]], plotcolor[::-1], [], tags)
    if peratom or plotatomic and not projectatomic:
        asap_plot.plot(proj_atomic_all[::-1, [pc1, pc2]], plotcolor_peratom[::-1],[],[])
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fmat', nargs='+', type=str, required=True,
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
    parser.add_argument('--sparsemode', type=str, default='fps', help='Sparsification method to use ([fps], [cur])')
    parser.add_argument('--n', type=int, default=0,
                        help='number of the representative samples, set negative if using no sparsification')
    parser.add_argument('--power', type=int, default=1,
                        help='Take the nth power of the polynomial kernel function. 1 means linear kernel.')
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
         args.keepraw, args.sparsemode, args.n, args.power, args.d, args.pc1, args.pc2, args.projectatomic, args.plotatomic, args.adjusttext)

