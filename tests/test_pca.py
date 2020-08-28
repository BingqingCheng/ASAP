#!/usr/bin/python3
import os

from matplotlib import pyplot as plt

from asaplib.data import ASAPXYZ
from asaplib.plot import Plotters, set_color_function
from asaplib.reducedim import Dimension_Reducers


def main():
    """

    Test if dimensionality reduction is working.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    prefix: string giving the filename prefix
    """
    fxyz = os.path.join(os.path.split(__file__)[0], 'small_molecules-SOAP.xyz')
    fmat = ['SOAP-n4-l3-c1.9-g0.23']
    fcolor = 'dft_formation_energy_per_atom_in_eV'
    pca_d = 10
    prefix = "test-dimensionality-reduction"
    foutput = prefix + "-pca-d" + str(pca_d)

    # try to read the xyz file
    asapxyz = ASAPXYZ(fxyz)
    desc, _ = asapxyz.get_descriptors(fmat, False)

    print(desc)
    """
    reduce_dict = { "pca": 
                   {"type": 'PCA', 'parameter':{"n_components": pca_d, "scalecenter": scale}}
                  }
    
    reduce_dict = {
                   "preprocessing": {"type": 'SCALE', 'parameter': None},
                   "umap":
                   {"type": 'UMAP', 'parameter':{"n_components": pca_d, "n_neighbors": 10}}
                  }    
    
    reduce_dict = {
        "reduce1_pca": {"type": 'PCA', 'parameter':{"n_components": 20, "scalecenter":True}},
        "reduce2_tsne": {"type": 'TSNE', 'parameter': {"n_components": 2, "perplexity":20}}
        }
    """

    reduce_dict = {
        "preprocessing": {"type": 'SCALE', 'parameter': None},
        "skpca":
            {"type": 'SPARSE_KPCA',
             'parameter': {"n_components": pca_d,
                           "kernel": {"first_kernel": {"type": 'linear', "normalize": True}}
                           }
             }
    }

    dreducer = Dimension_Reducers(reduce_dict)

    proj = dreducer.fit_transform(desc)

    # save
    asapxyz.set_descriptors(proj, 'pca_coord')
    asapxyz.write(foutput)

    # color scheme
    plotcolor, plotcolor_peratom, colorlabel, colorscale = set_color_function(fcolor, asapxyz)

    outfile = 'PCA_4_' + prefix + '-c-' + fcolor + '.png'

    fig_spec_dict = {
        'outfile': outfile,
        'show': False,
        'title': None,
        'xlabel': 'Principal Axis 1',
        'ylabel': 'Principal Axis 2',
        'xaxis': True, 'yaxis': True,
        'remove_tick': False,
        'rasterized': True,
        'fontsize': 16,
        'components': {
            "first_p": {"type": 'scatter', 'clabel': colorlabel},
            "second_p": {"type": 'annotate', 'adtext': False}
        }
    }
    asap_plot = Plotters(fig_spec_dict)
    asap_plot.plot(proj[::-1, [0, 1]], plotcolor[::-1], [], [])
    plt.show()


def test_gen(tmpdir):
    """Test the generation using pytest"""
    main()


if __name__ == '__main__':
    main()
