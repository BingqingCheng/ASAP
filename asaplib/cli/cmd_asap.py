"""
Module containing the top level asap command
"""
# import numpy as np
import warnings

warnings.filterwarnings("ignore")

from .func_asap import *
from .cmd_cli_options import *
from asaplib.io import ConvertStrToList


@click.group('asap')
@click.pass_context
def asap(ctx):
    """The command line tool of Automatic Selection And Prediction tools for materials and molecules (ASAP)"""
    # Configure, make sure we have a dict object
    ctx.ensure_object(dict)
    """ stores the input data, with keys ['fxyz'], ['stride'], [periodic] """
    ctx.obj['data'] = {}
    """ stores the ASAPXYZ object """
    ctx.obj['asapxyz'] = None
    """ stores a np matrix used as the design matrix """
    ctx.obj['designmatrix'] = None
    """ stores a Design_Matrix object """
    ctx.obj['dm'] = None
    """ stores the specifications of descriptors. i.e. {tag: {desc_spec}} """
    ctx.obj['descriptors'] = {}
    """stores the specifications of generating descriptor options"""
    ctx.obj['desc_options'] = {}
    """stores the specifications of clustering options"""
    ctx.obj['cluster_options'] = {}
    """stores the specifications of kde options"""
    ctx.obj['kde_options'] = {}
    """stores the specifications of fit options"""
    ctx.obj['fit_options'] = {}
    """stores the specifications of dimensionality reduction map"""
    ctx.obj['map_options'] = {}
    """stores the specifications of the output figure"""
    ctx.obj['fig_options'] = {}


@asap.command('select')
@click.option('--algorithm', '--algo', '-a',
              type=click.Choice(['random', 'fps', 'cur'], case_sensitive=False),
              help='Sparsification algorithm to use',
              show_default=True, default='fps')
@click.option('--nkeep', '-n', type=int,
              help='The number (int) or the ratio (float) of samples to keep.',
              show_default=False, default=1)
@click.option('--design_matrix', '-dm', cls=ConvertStrToList, default='[]',
              help='Location of descriptor matrix file or name of the tags in ase xyz file\
                 the type is a list  \'[dm1, dm2]\', \
                as we can put together simutanously several design matrix.')
@click.pass_context
@file_input_options
@file_output_options
@output_setup_options
def select(ctx, fxyz, design_matrix,
           algorithm, nkeep,
           prefix, savexyz, savetxt):
    """
    Select a subset of frames using sparsification algorithms
    """

    if not fxyz and not design_matrix[0]:
        return

    if prefix is None:
        prefix = "ASAP-select-" + algorithm + "-n-" + str(nkeep)
    ctx.obj['asapxyz'], ctx.obj['design_matrix'], _ = read_xyz_n_dm(fxyz, design_matrix, False, False, False)

    from asaplib.compressor import Sparsifier
    sparsifier = Sparsifier(algorithm)
    sbs = sparsifier.sparsify(ctx.obj['design_matrix'], nkeep)
    # save
    if savetxt:
        selection = np.zeros(asapxyz.get_num_frames(), dtype=int)
        for i in sbs:
            selection[i] = 1
        np.savetxt(prefix + '.index', selection, fmt='%d')
    if savexyz: ctx.obj['asapxyz'].write(prefix, sbs)


@asap.group('gen_desc')
@click.option('--stride', '-s',
              help='Read in the xyz trajectory with X stide. Default: read/compute all frames.',
              default=1)
@click.option('--periodic/--no-periodic',
              help='Is the system periodic? If not specified, will infer from the XYZ file.',
              default=True)
@click.pass_context
@state_input_options
@file_input_options
@file_input_format_options
@file_output_options
@para_options
def gen_desc(ctx, in_file, fxyz, fxyz_format, prefix, stride, periodic, number_processes):
    """
    Descriptor generation command
    This command function evaluated before the descriptor specific ones,
    we setup the general stuff here, such as read the files.
    """

    if not fxyz and not in_file:
        return

    if in_file:
        # Here goes the routine to compute the descriptors according to the
        # state file(s)
        ctx.obj.update(load_in_file(in_file))

    if prefix is None: prefix = "ASAP-desc"
    if fxyz is not None:
        ctx.obj['data']['fxyz'] = fxyz
        ctx.obj['data']['fxyz_format'] = fxyz_format
        ctx.obj['data']['stride'] = stride
        ctx.obj['data']['periodic'] = periodic
    ctx.obj['desc_options']['prefix'] = prefix
    ctx.obj['desc_options']['N_processes'] = number_processes


@gen_desc.command('soap')
@click.option('--cutoff', '-c', type=float,
              help='Cutoff radius',
              show_default=False, default=None)
@click.option('--nmax', '-n', type=int,
              help='Maximum radial label',
              show_default=False, default=None)
@click.option('--lmax', '-l', type=int,
              help='Maximum angular label (<= 9)',
              show_default=False, default=None)
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
              type=click.Choice(['none', 'smart', 'minimal', 'longrange'], case_sensitive=False),
              help='Try out our universal SOAP parameters.',
              show_default=True, default='minimal')
@click.pass_context
@desc_options
@atomic_to_global_desc_options
def soap(ctx, tag, cutoff, nmax, lmax, atom_gaussian_width, crossover, rbf, universal_soap,
         reducer_type, zeta, element_wise, peratom):
    """Generate SOAP descriptors"""
    # load up the xyz
    ctx.obj['asapxyz'] = load_asapxyz(ctx.obj['data'])

    if crossover is False:
        print("Warning: atomic species cross terms are not included! use --crossover if you want cross terms.")

    if universal_soap != 'none' and cutoff is None and nmax is None and lmax is None:
        from asaplib.hypers import universal_soap_hyper
        global_species = ctx.obj['asapxyz'].get_global_species()
        soap_spec = universal_soap_hyper(global_species, universal_soap, dump=True)
    elif cutoff is not None and nmax is not None and lmax is not None:
        soap_spec = {'soap1': {'type': 'SOAP',
                               'cutoff': cutoff,
                               'n': nmax,
                               'l': lmax,
                               'atom_gaussian_width': atom_gaussian_width}}
    else:
        raise IOError("Please either use universal soap or specify the values of cutoff, n, and l.")

    for k in soap_spec.keys():
        soap_spec[k]['rbf'] = rbf
        soap_spec[k]['crossover'] = crossover
    # The specification for the reducers
    reducer_spec = dict(set_reducer(reducer_type, element_wise, zeta))
    # The specification for the descriptor
    desc_spec = {}
    for k, v in soap_spec.items():
        desc_spec[k] = {'atomic_descriptor': dict({k: v}),
                        'reducer_function': reducer_spec}
    # specify descriptors using the cmd line tool
    ctx.obj['descriptors'][tag] = desc_spec
    # Compute the save the descriptors
    output_desc(ctx.obj['asapxyz'], ctx.obj['descriptors'], ctx.obj['desc_options']['prefix'], peratom,
                ctx.obj['desc_options']['N_processes'])


@gen_desc.command('acsf')
@click.option('--cutoff', '-c', type=float,
              help='Cutoff radius',
              show_default=False, default=None)
@click.option('--universal_acsf', '--uacsf', '-u',
              type=click.Choice(['none', 'smart', 'minimal', 'longrange'], case_sensitive=False),
              help='Try out our universal ACSF parameters.',
              show_default=True, default='minimal')
@click.pass_context
@desc_options
@atomic_to_global_desc_options
def acsf(ctx, tag, cutoff, universal_acsf,
         reducer_type, zeta, element_wise, peratom):
    """Generate ACSF descriptors"""
    # load up the xyz
    ctx.obj['asapxyz'] = load_asapxyz(ctx.obj['data'])

    from asaplib.hypers import universal_acsf_hyper
    global_species = ctx.obj['asapxyz'].get_global_species()
    if cutoff is not None:
        acsf_spec = universal_acsf_hyper(global_species, cutoff, dump=True, verbose=False)
    else:
        acsf_spec = universal_acsf_hyper(global_species, universal_acsf, dump=True, verbose=False)

    # The specification for the reducers
    reducer_spec = dict(set_reducer(reducer_type, element_wise, zeta))
    # The specification for the descriptor
    desc_spec = {'acsf': {'atomic_descriptor': acsf_spec,
                          'reducer_function': reducer_spec}
                 }
    # specify descriptors using the cmd line tool
    ctx.obj['descriptors'][tag] = desc_spec
    # Compute the save the descriptors
    output_desc(ctx.obj['asapxyz'], ctx.obj['descriptors'], ctx.obj['desc_options']['prefix'], peratom,
                ctx.obj['desc_options']['N_processes'])


@gen_desc.command('cm')
@click.pass_context
@desc_options
def cm(ctx, tag):
    """Generate the Coulomb Matrix descriptors"""
    # load up the xyz
    ctx.obj['asapxyz'] = load_asapxyz(ctx.obj['data'])
    # The specification for the descriptor
    ctx.obj['descriptors'][tag] = {'cm': {'type': "CM"}}
    # Compute the save the descriptors
    output_desc(ctx.obj['asapxyz'], ctx.obj['descriptors'], ctx.obj['desc_options']['prefix'], False,
                ctx.obj['desc_options']['N_processes'])


@gen_desc.command('run')
@click.pass_context
def run(ctx):
    """ Running analysis using input files """
    # load up the xyz
    ctx.obj['asapxyz'] = load_asapxyz(ctx.obj['data'])
    # Compute the save the descriptors
    output_desc(ctx.obj['asapxyz'], ctx.obj['descriptors'], ctx.obj['desc_options']['prefix'])


@asap.group('cluster', chain=True)
@click.pass_context
@file_input_options
@file_output_options
@dm_input_options
@km_input_options
@output_setup_options
def cluster(ctx, fxyz, design_matrix, use_atomic_descriptors, only_use_species, kernel_matrix,
            prefix, savexyz, savetxt):
    """
    Clustering using the design matrix.
    This command function evaluated before the specific ones,
    we setup the general stuff here, such as read the files.
    """

    if not fxyz and not design_matrix[0]:
        return
    if prefix is None: prefix = "ASAP-cluster"

    ctx.obj['cluster_options'] = {'prefix': prefix,
                                  'savexyz': savexyz,
                                  'savetxt': savetxt,
                                  'use_atomic_descriptors': use_atomic_descriptors,
                                  'only_use_species': only_use_species
                                  }

    ctx.obj['asapxyz'], ctx.obj['design_matrix'], _ = read_xyz_n_dm(fxyz, design_matrix, use_atomic_descriptors,
                                                                    only_use_species, False)

    if kernel_matrix != 'none':
        try:
            import numpy as np
            kNN = np.genfromtxt(kernel_matrix, dtype=float)
            print("loaded kernal matrix", kmat, "with shape", np.shape(kNN))
            from asaplib.kernel import kerneltodis
            ctx.obj['design_matrix'] = kerneltodis(kNN)
        except:
            raise ValueError('Cannot load the coordinates')


@cluster.command('fdb')
@click.pass_context
def fdb(ctx):
    """Clustering by fast search and find of density peaks (FDB)"""
    from asaplib.cluster import LAIO_DB
    trainer = LAIO_DB()

    ctx.obj['cluster_labels'] = cluster_process(ctx.obj['asapxyz'], trainer, ctx.obj['design_matrix'],
                                                ctx.obj['cluster_options'])


@cluster.command('dbscan')
@click.option('--metric', type=str,
              help='controls how distance is computed in the ambient space of the input data. \
                    See: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html',
              show_default=True, default='euclidean')
@click.option('--min_samples', '-ms', type=int,
              help='The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.',
              show_default=True, default=5)
@click.option('--eps', '-e', type=float,
              help='The maximum distance between two samples for one to be considered as in the neighborhood of the other.',
              default=None)
@click.pass_context
def dbscan(ctx, metric, min_samples, eps):
    """Density-based spatial clustering of applications with noise (DBSCAN)"""
    from asaplib.cluster import sklearn_DB
    if eps is None:
        from scipy.spatial.distance import cdist
        desc = ctx.obj['design_matrix']
        # we compute the characteristic bandwidth of the data
        # first select a subset of structures (20)
        import numpy as np
        sbs = np.random.choice(np.asarray(range(len(desc))), 50, replace=False)
        # the characteristic bandwidth of the data
        eps = np.percentile(cdist(desc[sbs], desc, metric), 100 * 10. / len(desc))

    trainer = sklearn_DB(eps, min_samples, metric)

    ctx.obj['cluster_labels'] = cluster_process(ctx.obj['asapxyz'], trainer, ctx.obj['design_matrix'],
                                                ctx.obj['cluster_options'])


@cluster.command('plot_pca')
@click.pass_context
@map_setup_options
@d_reduce_options
@file_output_options
def plot_pca(ctx, scale, dimension, axes,
             peratom, adjusttext, annotate, aspect_ratio, style, prefix):
    """ Plot the clustering results using a PCA map. Only use this command after fdb or dbscan.  """
    if prefix is None:
        prefix = "clustering-pca"
    colorlabel = "Clustering results"
    colorscale = [None, None]
    ctx.obj['map_options'].update({'color': ctx.obj['cluster_labels'],
                                   'color_atomic': [],
                                   'project_atomic': [],
                                   'only_use_species': ctx.obj['cluster_options']['only_use_species'],
                                   'peratom': False,
                                   'annotate': [],
                                   'outmode': 'none',
                                   'keepraw': True})
    if annotate != 'none':
        ctx.obj['map_options']['annotate'] = np.loadtxt(annotate, dtype="str")[:]

    ctx.obj['fig_options'] = figure_style_setups(prefix, colorlabel, colorscale, 'gnuplot', style, aspect_ratio,
                                                 adjusttext)
    ctx.obj['fig_options']['components'].update({"third_p": {"type": 'cluster', 'w_label': True, 'circle_size': 20}})
    map_name = "clustering-pca-d-" + str(dimension)
    reduce_dict = {'pca': {
        'type': 'PCA',
        'parameter': {"n_components": dimension, "scalecenter": scale}}
    }
    map_process(ctx.obj, reduce_dict, axes, map_name)


@asap.group('kde', chain=True)
@click.pass_context
@file_input_options
@file_output_options
@dm_input_options
@output_setup_options
def kde(ctx, fxyz, design_matrix, use_atomic_descriptors, only_use_species,
        prefix, savexyz, savetxt):
    """
    Kernel density estimation using the design matrix.
    This command function evaluated before the specific ones,
    we setup the general stuff here, such as read the files.
    """

    if not fxyz and design_matrix[0]:
        return
    if prefix is None: prefix = "ASAP-kde"

    ctx.obj['kde_options'] = {'prefix': prefix,
                              'savexyz': savexyz,
                              'savetxt': savetxt,
                              'use_atomic_descriptors': use_atomic_descriptors,
                              'only_use_species': only_use_species
                              }

    ctx.obj['asapxyz'], ctx.obj['design_matrix'], _ = read_xyz_n_dm(fxyz, design_matrix, use_atomic_descriptors,
                                                                    only_use_species, False)


@kde.command('kde_internal')
@click.option('--dimension', '-d', type=int,
              help='The number of the first D dimensions to keep when doing KDE.',
              show_default=True, default=8)
@click.pass_context
def kde_internal(ctx, dimension):
    """Internal implementation of KDE"""
    from asaplib.kde import KDE_internal
    import numpy as np
    proj = np.asmatrix(ctx.obj['design_matrix'])[:, 0:dimension]
    density_model = KDE_internal()
    ctx.obj['kde'] = kde_process(ctx.obj['asapxyz'], density_model, proj, ctx.obj['kde_options'])


@kde.command('kde_scipy')
@click.option('--dimension', '-d', type=int,
              help='The number of the first D dimensions to keep when doing KDE.',
              show_default=True, default=50)
@click.option('--bw_method', '-bw', type=str,
              help='This can be ‘scott’, ‘silverman’, a scalar constant or a callable.',
              show_default=False, default=None)
@click.pass_context
def kde_scipy(ctx, bw_method, dimension):
    """Scipy implementation of KDE"""
    from asaplib.kde import KDE_scipy
    import numpy as np
    proj = np.asmatrix(ctx.obj['design_matrix'])[:, 0:dimension]
    density_model = KDE_scipy(bw_method)
    ctx.obj['kde'] = kde_process(ctx.obj['asapxyz'], density_model, proj, ctx.obj['kde_options'])


@kde.command('kde_sklearn')
@click.option('--dimension', '-d', type=int,
              help='The number of the first D dimensions to keep when doing KDE.',
              show_default=True, default=50)
@click.option('--metric', type=str,
              help='controls how distance is computed in the ambient space of the input data. \
                    See: https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation',
              show_default=True, default='euclidean')
@click.option('--algorithm', type=click.Choice(['kd_tree', 'ball_tree', 'auto'], case_sensitive=False),
              help='Algorithm to use',
              show_default=True, default='auto')
@click.option('--kernel', type=click.Choice(['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'],
                                            case_sensitive=False),
              help='Kernel to use',
              show_default=True, default='gaussian')
@click.option('--bandwidth', '-bw', type=float,
              help='Bandwidth of the kernel',
              show_default=True, default=1)
@click.pass_context
def kde_scipy(ctx, dimension, bandwidth, algorithm, kernel, metric):
    """Scikit-learn implementation of KDE"""
    from asaplib.kde import KDE_sklearn
    import numpy as np
    proj = np.asmatrix(ctx.obj['design_matrix'])[:, 0:dimension]
    density_model = KDE_sklearn(bandwidth=bandwidth, algorithm=algorithm, kernel=kernel, metric=metric)
    ctx.obj['kde'] = kde_process(ctx.obj['asapxyz'], density_model, proj, ctx.obj['kde_options'])


@kde.command('plot_pca')
@click.pass_context
@map_setup_options
@d_reduce_options
@file_output_options
def plot_pca(ctx, scale, dimension, axes,
             peratom, adjusttext, annotate, aspect_ratio, style, prefix):
    """ Plot the KDE results using a PCA map """
    if prefix is None:
        prefix = "kde-pca"
    colorlabel = "Kernel density estimate (log scale)"
    colorscale = [None, None]
    ctx.obj['map_options'].update({'color': ctx.obj['kde'],
                                   'color_atomic': [],
                                   'project_atomic': [],
                                   'only_use_species': ctx.obj['kde_options']['only_use_species'],
                                   'peratom': False,
                                   'annotate': [],
                                   'outmode': 'none',
                                   'keepraw': True})
    if annotate != 'none':
        ctx.obj['map_options']['annotate'] = np.loadtxt(annotate, dtype="str")[:]

    ctx.obj['fig_options'] = figure_style_setups(prefix, colorlabel, colorscale, 'gnuplot', style, aspect_ratio,
                                                 adjusttext)
    map_name = "kde-pca-d-" + str(dimension)
    reduce_dict = {'pca': {
        'type': 'PCA',
        'parameter': {"n_components": dimension, "scalecenter": scale}}
    }
    map_process(ctx.obj, reduce_dict, axes, map_name)


@asap.group('map')
@click.pass_context
@file_input_options
@file_output_options
@dm_input_options
@map_setup_options
@map_io_options
@color_setup_options
def map(ctx, fxyz, design_matrix, prefix, output, extra_properties,
        use_atomic_descriptors, only_use_species, peratom, keepraw,
        color, color_column, color_label, colormap, color_from_zero, normalized_by_size,
        annotate, adjusttext, style, aspect_ratio):
    """
    Making 2D maps using dimensionality reduction.
    This command function evaluated before the specific ones,
    we setup the general stuff here, such as read the files.
    """

    if not fxyz and not design_matrix[0]:
        return
    if prefix is None: prefix = "ASAP-lowD-map"
    ctx.obj['asapxyz'], ctx.obj['design_matrix'], ctx.obj['design_matrix_atomic'] = read_xyz_n_dm(fxyz, design_matrix,
                                                                                                  use_atomic_descriptors,
                                                                                                  only_use_species,
                                                                                                  peratom)

    # Read additional properties
    if extra_properties:
        ctx.obj['asapxyz'].load_properties(extra_properties)

    if ctx.obj['asapxyz'] is None: output = 'matrix'
    print(len(ctx.obj['design_matrix_atomic']))
    # remove the raw descriptors
    if not keepraw and ctx.obj['asapxyz'] is not None:
        print("Remove raw desciptors..")
        ctx.obj['asapxyz'].remove_descriptors(design_matrix)
        ctx.obj['asapxyz'].remove_atomic_descriptors(design_matrix)

    # color scheme
    from asaplib.plot import set_color_function
    plotcolor, plotcolor_peratom, colorlabel, colorscale = set_color_function(color, ctx.obj['asapxyz'], color_column,
                                                                              0, peratom, use_atomic_descriptors,
                                                                              only_use_species, color_from_zero,
                                                                              normalized_by_size)
    if color_label is not None: colorlabel = color_label

    ctx.obj['map_options'] = {'color': plotcolor,
                              'color_atomic': plotcolor_peratom,
                              'project_atomic': use_atomic_descriptors,
                              'only_use_species': only_use_species,
                              'peratom': peratom,
                              'annotate': [],
                              'outmode': output,
                              'keepraw': keepraw
                              }
    if annotate != 'none':
        try:
            ctx.obj['map_options']['annotate'] = ctx.obj['asapxyz'].get_property(annotate)
        except:
            import numpy as np
            ctx.obj['map_options']['annotate'] = np.loadtxt(annotate, dtype="str")[:]

    ctx.obj['fig_options'] = figure_style_setups(prefix, colorlabel, colorscale, colormap, style, aspect_ratio,
                                                 adjusttext)


@map.command('raw')
@click.pass_context
@d_reduce_options
def raw(ctx, scale, dimension, axes):
    """Just plot the raw coordinates"""
    map_name = "raw-d-" + str(dimension)
    reduce_dict = {
        'type': 'RAW',
        'parameter': {"n_components": dimension, "scalecenter": scale}}
    map_process(ctx.obj, reduce_dict, axes, map_name)


@map.command('pca')
@click.pass_context
@d_reduce_options
def pca(ctx, scale, dimension, axes):
    """Principal Component Analysis"""
    map_name = "pca-d-" + str(dimension)
    reduce_dict = {'pca': {
        'type': 'PCA',
        'parameter': {"n_components": dimension, "scalecenter": scale}}
    }
    if scale:
        print("Perform standard scaling of the design matrix. To turn it off use `--no-scale`")
    map_process(ctx.obj, reduce_dict, axes, map_name)


@map.command('skpca')
@click.pass_context
@d_reduce_options
@kernel_options
@sparsification_options
def skpca(ctx, scale, dimension, axes,
          kernel, kernel_parameter, sparse_mode, n_sparse):
    """Sparse Kernel Principal Component Analysis"""
    map_name = "skpca-d-" + str(dimension)
    reduce_dict = {}
    if scale:
        print("Perform standard scaling of the design matrix. To turn it off use `--no-scale`")
        reduce_dict = {"preprocessing": {"type": 'SCALE', 'parameter': None}}
    reduce_dict['skpca'] = {"type": 'SPARSE_KPCA',
                            'parameter': {"n_components": dimension,
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
    map_name = "umap-d-" + str(dimension)
    reduce_dict = {}
    if scale:
        print("Perform standard scaling of the design matrix. To turn it off use `--no-scale`")
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
    map_name = "tsne-d-" + str(dimension)
    reduce_dict = {}
    if pca:
        # pre-process with PCA if dim > 50
        # suggested here: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        reduce_dict = {"preprocessing": {"type": 'PCA', 'parameter': {"n_components": 50, "scalecenter": scale}}}
    elif scale:
        print("Perform standard scaling of the design matrix. To turn it off use `--no-scale`")
        reduce_dict = {"preprocessing": {"type": 'SCALE', 'parameter': None}}
    reduce_dict['tsne'] = {'type': 'TSNE', 'parameter':
        {'perplexity': perplexity,
         'early_exaggeration': early_exaggeration,
         'learning_rate': learning_rate,
         'metric': metric
         }}
    map_process(ctx.obj, reduce_dict, axes, map_name)


@asap.group('fit')
@click.pass_context
@file_input_options
@file_output_options
@dm_input_options
@fit_setup_options
def fit(ctx, fxyz, design_matrix, use_atomic_descriptors, only_use_species, y, normalized_by_size, prefix,
        test_ratio, learning_curve, lc_points):
    """
    Fit a machine learning model to the design matrix and labels.
    This command function evaluated before the specific ones,
    we setup the general stuff here, such as read the files.
    """
    if not fxyz and not design_matrix[0]:
        return
    if prefix is None: prefix = "ASAP-fit"

    ctx.obj['fit_options'] = {"prefix": prefix,
                              "learning_curve": learning_curve,
                              "lc_points": lc_points,
                              "test_ratio": test_ratio
                              }
    asapxyz, desc, _ = read_xyz_n_dm(fxyz, design_matrix, use_atomic_descriptors, only_use_species, False)

    try:
        import numpy as np
        y_all = np.genfromtxt(y, dtype=float)
    except:
        if use_atomic_descriptors:
            y_all = asapxyz.get_atomic_property(y, normalized_by_size)
        else:
            y_all = asapxyz.get_property(y, normalized_by_size)
    # print(y_all)

    from asaplib.data import Design_Matrix
    ctx.obj['dm'] = Design_Matrix(desc, y_all, True, test_ratio)


@fit.command('ridge')
@click.option('--sigma', type=float,
              help='the noise level of the signal. Also the regularizer that improves the stablity of matrix inversion.',
              default=0.0001)
@click.pass_context
def ridge(ctx, sigma):
    """Ridge Regression"""
    from asaplib.fit import RidgeRegression
    rr = RidgeRegression(sigma)
    # fit the model
    ctx.obj['dm'].compute_fit(rr, 'ridge_regression', store_results=True, plot=True)
    if ctx.obj['fit_options']["learning_curve"] > 1:
        ctx.obj['dm'].compute_learning_curve(rr, 'ridge_regression', ctx.obj['fit_options']["learning_curve"],
                                             ctx.obj['fit_options']["lc_points"], randomseed=42, verbose=False)

    ctx.obj['dm'].save_state(ctx.obj['fit_options']['prefix'])
    from matplotlib import pyplot as plt
    plt.show()


@fit.command('kernelridge')
@click.option('--sigma', type=float,
              help='the noise level of the signal. Also the regularizer that improves the stablity of matrix inversion.',
              default=0.0001)
@kernel_options
@sparsification_options
@click.pass_context
def kernelridge(ctx, sigma, kernel, kernel_parameter, sparse_mode, n_sparse):
    """Kernel Ridge Regression (with sparsification)"""
    from asaplib.fit import SPARSE_KRR_Wrapper, KRRSparse
    k_spec = {"first_kernel": {"type": kernel, "d": kernel_parameter}}
    krr = KRRSparse(0., None, sigma)
    skrr = SPARSE_KRR_Wrapper(k_spec, krr, sparse_mode=sparse_mode, n_sparse=n_sparse)
    # fit the model
    ctx.obj['dm'].compute_fit(skrr, 'skrr', store_results=True, plot=True)
    if ctx.obj['fit_options']["learning_curve"] > 1:
        ctx.obj['dm'].compute_learning_curve(skrr, 'skrr', ctx.obj['fit_options']["learning_curve"],
                                             ctx.obj['fit_options']["lc_points"], randomseed=42, verbose=False)

    ctx.obj['dm'].save_state(ctx.obj['fit_options']['prefix'])
    from matplotlib import pyplot as plt
    plt.show()
