"""
Module containing the top level asap command
"""
import os
import json
from yaml import full_load as yload
import numpy as np
import click
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from .func_asap import *
from asaplib.data import ASAPXYZ, Design_Matrix
from asaplib.reducedim import Dimension_Reducers
from asaplib.plot import Plotters, set_color_function
from asaplib.io import ConvertStrToList

@click.group('asap')
@click.pass_context
def asap(ctx):
    """The command line tool of Automatic Selection And Prediction tools for materials and molecules (ASAP)"""
    # Configure, make sure we have a dict object
    ctx.ensure_object(dict)


def desc_options(f):
    """Create common options for a descriptor command"""
    f = click.option('--peratom', 
                     help='Save the per-atom local descriptors.',
                     show_default=True, default=False, is_flag=True)(f)
    f = click.option('--tag',
                     help='Tag for the descriptor output.',
                     default='ASAP')(f)
    f = click.option('--prefix', '-p',
                     help='Prefix to be used for the output XYZ file.', 
                     default='ASAP-desc')(f)
    f = click.option('--periodic/--no-periodic', 
                     help='Is the system periodic? If not specified, will infer from the XYZ file.',
                     default=True)(f)
    f = click.option('--stride', '-s',
                     help='Read in the xyz trajectory with X stide. Default: read/compute all frames.',
                     default=1)(f)
    f = click.option('--fxyz', '-f', 
                     type=click.Path('r'), 
                     help='Input XYZ file',
                     default='ASAP.xyz')(f)
    f = click.option('--in_file', '--in', '-i', type=click.Path('r'),
                     help='The state file that includes a dictionary-like specifications of descriptors to use.')(f)
    return f

def atomic_to_global_desc_options(f):
    """Create common options for global descriptors constructed based on atomic fingerprints """
    f = click.option('--kernel_type', '-k',
                     help='type of operations to get global descriptors from the atomic soap vectors, e.g. \
                          [average], [sum], [moment_avg], [moment_sum].',
                     show_default=True, default='average', type=str)(f)
    f = click.option('--zeta', '-z', 
                     help='Moments to take when converting atomic descriptors to global ones.',
                     default=1, type=int)(f)
    f = click.option('--element_wise', '-e', 
                     help='element-wise operation to get global descriptors from the atomic soap vectors',
                     show_default=True, default=False, is_flag=True)(f)
    return f

@asap.group('gen_desc')
@click.pass_context
@desc_options
def gen_desc(ctx, in_file, fxyz, prefix, tag, 
             peratom, stride, periodic):
    """
    Descriptor generation command
    This command function evaluated before the descriptor specific ones,
    we setup the general stuff here, such as read the files.
    """

    if in_file:
        # Here goes the routine to compute the descriptors according to the
        # state file(s)
        # TODO: write a proper parser function!
        with open(in_file, 'r') as stream:
            try:
                state_str = yload(stream)
            except:
                state_str = json.load(stream)
        state = {}
        for k,s in state_str.items():
            if isinstance(s, str):
                state[k] = json.loads(s)
            else:
                state[k] = s
                
        if fxyz == None:
           fxyz = state['data']['fxyz']
        ctx.obj['desc_spec'] = state['descriptors']

    ctx.obj['asapxyz'] = ASAPXYZ(fxyz, stride, periodic)
    ctx.obj['desc_settings'] = {
        'tag': tag,
        'prefix': prefix,
        'peratom': peratom,
    }

@gen_desc.command('soap')
@click.option('--cutoff', '-c', type=float, 
              help='Cutoff radius', 
              show_default=True, default=3.0)
@click.option('--nmax', '-n', type=int, 
              help='Maximum radial label', 
              show_default=True, default=6)
@click.option('--lmax', '-l', type=int, 
              help='Maximum angular label (<= 9)', 
              show_default=True, default=6)
@click.option('--rbf', type=click.Choice(['gto', 'polynomial'], case_sensitive=False), 
              help='Radial basis function', 
              show_default=True, default='gto')
@click.option('--atom-gaussian-width', '-sigma', '-g', type=float, 
              help='The width of the Gaussian centered on atoms.', 
              show_default=True, default=0.5)
@click.option('--crossover/--no-crossover', 
              help='If to included the crossover of atomic types.', 
              show_default=True, default=False)
@click.option('--universal_soap', '--usoap', '-u',
              type=click.Choice(['none','smart','minimal', 'longrange'], case_sensitive=False), 
              help='Try out our universal SOAP parameters.', 
              show_default=True, default='none')
@click.pass_context
@atomic_to_global_desc_options
def soap(ctx, cutoff, nmax, lmax, atom_gaussian_width, crossover, rbf,
         kernel_type, zeta, element_wise, universal_soap):
    """Generate SOAP descriptors"""
    if universal_soap != 'none':
        from asaplib.hypers import universal_soap_hyper
        global_species = ctx.obj['asapxyz'].get_global_species()
        soap_spec = universal_soap_hyper(global_species, universal_soap, dump=True)
        for k in soap_spec.keys():
            soap_spec[k]['rbf'] = rbf
            soap_spec[k]['crossover'] = crossover
    else:
        soap_spec = {
            'soap1': {
                'type': 'SOAP',
                'cutoff': cutoff,
                'n': nmax,
                'l': lmax,
                'atom_gaussian_width': atom_gaussian_width,
                'rbf': rbf,
                'crossover': crossover
            }
        }
    # The specification for the kernels
    kernel_spec = dict(set_kernel(kernel_type, element_wise, zeta))
    # The specification for the descriptor
    desc_spec = {}
    for k, v in soap_spec.items():
        desc_spec[k] = {
                'atomic_descriptor': dict({k: v}),
                'kernel_function': kernel_spec}

    # Compute the save the descriptors
    output_desc(ctx.obj['asapxyz'], desc_spec, ctx.obj['desc_settings'])

@gen_desc.command('cm')
@click.pass_context
def cm(ctx):
    """Generate the Coulomb Matrix descriptors"""
    # The specification for the descriptor
    desc_spec = {'cm': {'type': "CM"}}
    # Compute the save the descriptors
    output_desc(ctx.obj['asapxyz'], desc_spec, ctx.obj['desc_settings'])

@gen_desc.command('run')
@click.pass_context
def run(ctx):
    """ Running analysis using input files """
    output_desc(ctx.obj['asapxyz'], ctx.obj['desc_spec'], ctx.obj['desc_settings'])

def map_setup_options(f):
    """Create common options for making 2D maps of the data set"""
    f = click.option('--adjusttext/--no-adjusttext', 
                     help='Adjust the annotation texts so they do not overlap.',
                     default=False)(f)
    f = click.option('--keepraw/--no-keepraw', 
                     help='Keep the high dimensional descriptor when output XYZ file.',
                     default=False)(f)
    f = click.option('--peratom', 
                     help='Save the per-atom local descriptors.',
                     default=False, is_flag=True)(f)
    f = click.option('--project_atomic', 
                     help='Build map based on atomic descriptors instead of global ones.',
                     default=False, is_flag=True)(f)
    f = click.option('--output', '-o', type=click.Choice(['xyz', 'matrix'], case_sensitive=False), 
                     help='Output file format.',
                     default='xyz')(f)
    f = click.option('--prefix', '-p', 
                      help='Prefix for the output png.', default='ASAP-lowD')(f)
    f = click.option('--annotate', '-a',
                     help='Location of tags to annotate the samples.',
                     default='none', type=str)(f)
    f = click.option('--color_column', '-ccol',
                     help='The column number used in the color file. Starts from 0.',
                     default=0)(f)
    f = click.option('--color', '-c',
                     help='Location of a file or name of the properties in the XYZ file. \
                     Used to color the scatter plot for all samples (N floats).',
                     default='none', type=str)(f)
    f = click.option('--fxyz', '-f', 
                     type=click.Path('r'), 
                     help='Input XYZ file',
                     default='ASAP.xyz')(f)
    f = click.option('--design_matrix', '-dm', cls=ConvertStrToList, default=[],
                     help='Location of descriptor matrix file or name of the tags in ase xyz file\
                           the type is a list  \'[dm1, dm2]\', as we can put together simutanously several design matrix.')(f)
    f = click.option('--in_file', '--in', '-i', type=click.Path('r'),
                     help='The state file that includes a dictionary-like specifications of descriptors to use.')(f)
    return f

@asap.group('map')
@click.pass_context
@map_setup_options
def map(ctx, in_file, fxyz, design_matrix, prefix, output,
         project_atomic, peratom, keepraw,
         color, color_column,
         annotate, adjusttext):
    """
    Making 2D maps using dimensionality reduction.
    This command function evaluated before the specific ones,
    we setup the general stuff here, such as read the files.
    """

    if in_file:
        print('')
        # Here goes the routine to compute the descriptors according to the
        # state file(s)

    ctx.obj['asapxyz'], ctx.obj['design_matrix'], ctx.obj['design_matrix_atomic'] = read_xyz_n_dm(fxyz, design_matrix, project_atomic, peratom)
    if ctx.obj['asapxyz'] is None: output = 'matrix'

    # remove the raw descriptors
    if not keepraw:
        ctx.obj['asapxyz'].remove_descriptors(design_matrix)
        ctx.obj['asapxyz'].remove_atomic_descriptors(design_matrix)

    # color scheme
    plotcolor, plotcolor_peratom, colorlabel, colorscale = set_color_function(color, ctx.obj['asapxyz'], color_column, 0, peratom, project_atomic)
    ctx.obj['map_info'] =  { 'color': plotcolor, 
                              'color_atomic': plotcolor_peratom,
                              'peratom': peratom,
                              'annotate': [],
                              'outmode': output,
                              'keepraw': keepraw
                           }
    if annotate != 'none':
        ctx.obj['plot_info']['annotate'] = np.loadtxt(annotate, dtype="str")[:] 

    ctx.obj['fig_spec_dict'] = {
        'outfile': prefix,
        'show': False,
        'title': None,
        'xlabel': 'Principal Axis 1',
        'ylabel': 'Principal Axis 2',
        'xaxis': True,  'yaxis': True,
        'remove_tick': False,
        'rasterized': True,
        'fontsize': 16,
        'components':{ 
            "first_p": {"type": 'scatter', 'clabel': colorlabel, 
                        'vmin': colorscale[0], 'vmax': colorscale[0]},
            "second_p": {"type": 'annotate', 'adtext': adjusttext}
             }
        }

def d_reduce_options(f):
    """Create common options for dimensionality reduction"""
    f = click.option('--axes', nargs=2, type=click.Tuple([int, int]),
                     help='Plot the projection along which projection axes.',
                     default=[0,1])(f)
    f = click.option('--dimension', '-d',
                     help='Number of the dimensions to keep in the output XYZ file.',
                     default=10)(f)
    f = click.option('--scale/--no-scale', 
                     help='Standard scaling of the coordinates. Recommended.',
                     default=True)(f)
    return f

@map.command('pca')
@click.pass_context
@d_reduce_options
def pca(ctx, scale, dimension, axes):
    """Principal Component Analysis"""
    map_name = "pca-d-"+str(dimension)
    reduce_dict = {'pca': {
                   'type': 'PCA', 
                   'parameter':{"n_components": dimension, "scalecenter": scale}}
                  }
    map_process(ctx.obj, reduce_dict, axes, map_name)

@map.command('skpca')
@click.option('--n_sparse', '-n', type=int, 
              help='number of the representative samples, set negative if using no sparsification', 
              show_default=True, default=10)
@click.option('--sparse_mode', '-s',
              type=click.Choice(['random', 'cur', 'fps'], case_sensitive=False), 
              help='Sparsification method to use.', 
              show_default=True, default='fps')
@click.option('--kernel_parameter', '-kp', type=float, 
              help='Parameter used in the kernel function.', 
              default=None)
@click.option('--kernel', '-k',
              type=click.Choice(['linear', 'polynomial', 'cosine'], case_sensitive=False), 
              help='Kernel function for converting design matrix to kernel matrix.', 
              show_default=True, default='linear')
@click.pass_context
@d_reduce_options
def skpca(ctx, scale, dimension, axes, 
          kernel, kernel_parameter, sparse_mode, n_sparse):
    """Sparse Kernel Principal Component Analysis"""
    map_name = "skpca-d-"+str(dimension)
    if scale:
        reduce_dict = {"preprocessing": {"type": 'SCALE', 'parameter': None}}
    reduce_dict['skpca'] = {"type": 'SPARSE_KPCA', 
                            'parameter':{"n_components": dimension, 
                                         "sparse_mode": sparse_mode, "n_sparse": n_sparse,
                                "kernel": {"first_kernel": {"type": kernel, "d": kernel_parameter}}}}
    map_process(ctx.obj, reduce_dict, axes, map_name)

@map.command('umap')
@click.option('--n_neighbors', '-nn', type=int, 
              help='Controls how UMAP balances local versus global structure in the data.', 
              show_default=True, default=10)
@click.option('--min_dist', '-md', type=float, 
              help='controls how tightly UMAP is allowed to pack points together.', 
              show_default=True, default=0.1)
@click.option('--metric', type=str, 
              help='controls how distance is computed in the ambient space of the input data. \
                    See: https://umap-learn.readthedocs.io/en/latest/parameters.html#metric', 
              show_default=True, default='euclidean')
@click.pass_context
@d_reduce_options
def umap(ctx, scale, dimension, axes, n_neighbors, min_dist, metric):
    """UMAP"""
    map_name = "umap-d-"+str(dimension)
    if scale:
        reduce_dict = {"preprocessing": {"type": 'SCALE', 'parameter': None}}
    reduce_dict['umap'] = {'type': 'UMAP', 'parameter':
                           {'n_components': dimension, 
                            'n_neighbors': n_neighbors,
                            'min_dist': min_dist,
                            'metric': metric
                          }}
    map_process(ctx.obj, reduce_dict, axes, map_name)

@map.command('tsne')
@click.option('--metric', type=str, 
              help='controls how distance is computed in the ambient space of the input data. \
                    See: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html', 
              show_default=True, default='euclidean')
@click.option('--learning_rate', '-l', type=float, 
              help='The learning rate is usually in the range [10.0, 1000.0].', 
              show_default=True, default=200.0)
@click.option('--early_exaggeration', '-e', type=float, 
              help='Controls how tight natural clusters in the original space are in the embedded space.', 
              show_default=True, default=12.0)
@click.option('--perplexity', '-p', type=float, 
              help='The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. \
                  Larger datasets usually require a larger perplexity. \
               Consider selecting a value between 5 and 50. \
               Different values can result in significanlty different results.', 
              show_default=True, default=30.0)
@click.option('--pca/--no-pca', 
                     help='Preprocessing the data using PCA with dimension 50. Recommended.',
                     default=True)
@click.pass_context
@d_reduce_options
def tsne(ctx, pca, scale, dimension, axes, 
          perplexity, early_exaggeration, learning_rate, metric):
    """t-SNE"""
    map_name = "tsne-d-"+str(dimension)

    if pca:
        # pre-process with PCA if dim > 50
        # suggested here: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        reduce_dict= {"preprocessing":{"type": 'PCA', 'parameter':{"n_components": 50, "scalecenter": scale}}}
    elif scale:
        reduce_dict = {"preprocessing": {"type": 'SCALE', 'parameter': None}}
    reduce_dict['tsne'] = {'type': 'TSNE', 'parameter':
                           {'perplexity': perplexity, 
                            'early_exaggeration': early_exaggeration,
                            'learning_rate': learning_rate,
                            'metric': metric
                          }}
    map_process(ctx.obj, reduce_dict, axes, map_name)

def fit_setup_options(f):
    """Create common options for making 2D maps of the data set"""
    f = click.option('--test_ratio', '--test', '-t', type=float, 
              help='Test ratio.', 
              show_default=True, default=0.05)(f)
    f = click.option('--prefix', '-p', 
                      help='Prefix for the output.', default='ASAP-fit')(f)
    f = click.option('--y', '-y',
                     help='Location of a file or name of the properties in the XYZ file',
                     default='none', type=str)(f)
    f = click.option('--fxyz', '-f', 
                     type=click.Path('r'), 
                     help='Input XYZ file',
                     default='ASAP.xyz')(f)
    f = click.option('--design_matrix', '-dm', cls=ConvertStrToList, default=[],
                     help='Location of descriptor matrix file or name of the tags in ase xyz file\
                           the type is a list  \'[dm1, dm2]\', as we can put together simutanously several design matrix.')(f)
    f = click.option('--in_file', '--in', '-i', type=click.Path('r'),
                     help='The state file that includes a dictionary-like specifications of fits to use.')(f)
    return f

@asap.group('fit')
@click.pass_context
@fit_setup_options
def fit(ctx, in_file, fxyz, design_matrix, y, prefix, test_ratio):
    """
    Fit a machine learning model to the design matrix and labels.
    This command function evaluated before the specific ones,
    we setup the general stuff here, such as read the files.
    """
    ctx.obj['fit_prefix'] = prefix
    if in_file:
        print('')
        # Here goes the routine to compute the descriptors according to the
        # state file(s)

    asapxyz, desc, _ = read_xyz_n_dm(fxyz, design_matrix, False, False)

    try:
        y_all = np.genfromtxt(y, dtype=float)
    except:
        y_all = asapxyz.get_property(y)

    ctx.obj['dm'] = Design_Matrix(desc, y_all, True, test_ratio)

@fit.command('ridge')
@click.option('--sigma', '-s', type=float, 
              help='the noise level of the signal. Also the regularizer that improves the stablity of matrix inversion.', 
              default=0.001)
@click.pass_context
def ridge(ctx, sigma):
    """Ridge Regression"""
    from asaplib.fit import RidgeRegression
    rr = RidgeRegression(sigma)
    # fit the model
    ctx.obj['dm'].compute_fit(rr, 'ridge_regression', store_results=True, plot=True)
    plt.show()

