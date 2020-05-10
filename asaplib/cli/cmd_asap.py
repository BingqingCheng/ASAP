"""
Module containing the top level asap command
"""
import os
import json
from yaml import full_load as yload
import numpy as np
import click
from matplotlib import pyplot as plt

from asaplib.data import ASAPXYZ
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
                     default=False, is_flag=True)(f)
    f = click.option('--tag',
                     help='Tag for the descriptor output.',
                     default='ASAP')(f)
    f = click.option('--prefix', '--p',
                     help='Prefix to be used for the output XYZ file.', 
                     default='ASAP')(f)
    f = click.option('--periodic/--no-periodic', 
                     help='Is the system periodic? If not specified, will infer from the XYZ file.',
                     default=True)(f)
    f = click.option('--stride', '--s',
                     help='Read in the xyz trajectory with X stide. Default: read/compute all frames.',
                     default=1)(f)
    f = click.option('--fxyz', '--f', 
                     type=click.Path('r'), help='Input XYZ file')(f)
    f = click.option('--in_file', '--in', '-i', type=click.Path('r'),
                     help='The state file that includes a dictionary-like specifications of descriptors to use.')(f)
    return f

def atomic_to_global_desc_options(f):
    """Create common options for global descriptors constructed based on atomic fingerprints """
    f = click.option('--kernel_type', '--k',
                     help='type of operations to get global descriptors from the atomic soap vectors, e.g. \
                          [average], [sum], [moment_avg], [moment_sum].',
                     default='average', type=str)(f)
    f = click.option('--zeta', '--z', 
                     help='Moments to take when converting atomic descriptors to global ones.',
                     default=1, type=int)(f)
    f = click.option('--element_wise', '--e', 
                     help='element-wise operation to get global descriptors from the atomic soap vectors',
                     default=False, is_flag=True)(f)
    return f

def set_kernel(kernel_type, element_wise, zeta):
    """
    setting up the kernel function that is used to convert atomic descriptors into global descriptors for a structure.
    At the moment only one single kernel function can be used.
    """
    kernel_func = {'kernel': {
        'kernel_type': kernel_type, # [average], [sum], [moment_average], [moment_sum]
        'element_wise': element_wise,
    }}
    if kernel_type == 'moment_average' or kernel_type == 'moment_sum':
        kernel_func['zeta'] = zeta
    return kernel_func

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
@click.option('--universal_soap', '--usoap', '--u',
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

def output_desc(asapxyz, desc_spec, desc_settings):
    """
    Compute and save the descritptors
    """
    # compute the descripitors
    tag = desc_settings['tag']
    peratom = desc_settings['peratom']
    prefix = desc_settings['prefix']
    asapxyz.compute_global_descriptors(desc_spec_dict=desc_spec,
                                       sbs=[],
                                       keep_atomic=peratom,
                                       tag=tag)
    asapxyz.write(prefix)
    asapxyz.save_state(tag)

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
    f = click.option('--output', '--o', type=click.Choice(['xyz', 'matrix'], case_sensitive=False), 
                     help='Output file format.',
                     default='xyz')(f)
    f = click.option('--prefix', '--p', 
                      help='Prefix for the output png.', default='ASAP')(f)
    f = click.option('--annotate', '--a',
                     help='Location of tags to annotate the samples.',
                     default='none', type=str)(f)
    f = click.option('--color_column',
                     help='The column number used in the color file. Starts from 0.',
                     default=0)(f)
    f = click.option('--color', "--c",
                     help='Location of a file or name of the properties in the XYZ file. \
                     Used to color the scatter plot for all samples (N floats).',
                     default='none', type=str)(f)
    f = click.option('--fxyz', '--f', 
                     type=click.Path('r'), 
                     help='Location of descriptor matrix file or name of the descriptors in ase xyz file.')(f)
    f = click.option('--design_matrix', '--dm', cls=ConvertStrToList, default=[],
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

    # try to read the xyz file
    if fxyz != 'none':
        ctx.obj['asapxyz'] = ASAPXYZ(fxyz)
        if project_atomic:
            _, ctx.obj['design_matrix'] = asapxyz.get_descriptors(design_matrix, True)
        else:
            ctx.obj['design_matrix'], ctx.obj['design_matrix_atomic'] = ctx.obj['asapxyz'].get_descriptors(design_matrix, peratom)
    else:
        ctx.obj['asapxyz'] = None
        print("Did not provide the xyz file. We can only output descriptor matrix.")
        output = 'matrix'
    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(design_matrix[0]):
        try:
            ctx.obj['design_matrix'] = np.genfromtxt(design_matrix[0], dtype=float)
            print("loaded the descriptor matrix from file: ", design_matrix[0])
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    #print(ctx.obj['design_matrix'])

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
    f = click.option('--dimension', '--d',
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
    from asaplib.pca import PCA
    map_name = "pca-d-"+str(dimension)

    pca = PCA(dimension, scale)
    proj = pca.fit_transform(ctx.obj['design_matrix'])
    if ctx.obj['map_info']['peratom']:
        proj_atomic = pca.transform(ctx.obj['design_matrix_atomic'])
    else:
        proj_atomic = None

    fig_spec = ctx.obj['fig_spec_dict']
    plotcolor = ctx.obj['map_info']['color']
    plotcolor_atomic = ctx.obj['map_info']['color_atomic']
    annotate = ctx.obj['map_info']['annotate']
    # plot
    map_plot(fig_spec, proj, proj_atomic, plotcolor, plotcolor_atomic, annotate, axes, map_name)
    # output 
    outfilename = ctx.obj['fig_spec_dict']['outfile']
    outmode =  ctx.obj['fig_spec_dict']['outmode']
    map_save(outfilename, outmode, ctx.obj['asapxyz'], proj, proj_atomic)

@map.command('umap')
@click.pass_context
@d_reduce_options
def umap(ctx, scale, dimension, axes):
    """UMAP"""
    from umap import UMAP
    map_name = "umap-d-"+str(dimension)

    # scale & center
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        desc = scaler.fit_transform(ctx.obj['design_matrix'])  # normalizing the features
    else:
        desc = ctx.obj['design_matrix']

    reducer = UMAP()
    proj = reducer.fit_transform(desc)
    if ctx.obj['map_info']['peratom']:
        proj_atomic = reducer.transform(ctx.obj['design_matrix_atomic'])
    else:
        proj_atomic = None

    fig_spec = ctx.obj['fig_spec_dict']
    plotcolor = ctx.obj['map_info']['color']
    plotcolor_atomic = ctx.obj['map_info']['color_atomic']
    annotate = ctx.obj['map_info']['annotate']
    # plot
    map_plot(fig_spec, proj, proj_atomic, plotcolor, plotcolor_atomic, annotate, axes, map_name)
    # output 
    outfilename = ctx.obj['fig_spec_dict']['outfile']
    outmode =  ctx.obj['fig_spec_dict']['outmode']
    map_save(outfilename, outmode, ctx.obj['asapxyz'], proj, proj_atomic)

def map_plot(fig_spec, proj, proj_atomic, plotcolor, plotcolor_atomic, annotate, axes, map_name):
    """
    Make plots
    """
    asap_plot = Plotters(fig_spec)
    asap_plot.plot(proj[::-1, axes], plotcolor[::-1], [], annotate)
    if proj_atomic is not None:
        asap_plot.plot(proj_atomic[::-1, axes], plotcolor_atomic[::-1],[],[])
    plt.show()

def map_save(foutput, outmode, asapxyz, proj, proj_atomic):
    """
    Save the low-D projections
    """
    if outmode == 'matrix':
        np.savetxt(foutput + ".coord", proj, fmt='%4.8f', header='low D coordinates of samples')
        if proj_atomic is not None:
            np.savetxt(foutput + "-atomic.coord", proj_atomic_all, fmt='%4.8f', header=map_name)
    if outmode == 'xyz':
        asapxyz.set_descriptors(proj, 'pca_coord')
        if proj_atomic is not None:
            asapxyz.set_atomic_descriptors(proj_atomic_all, map_name)
        asapxyz.write(foutput)
