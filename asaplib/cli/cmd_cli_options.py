import click

from asaplib.io import ConvertStrToList


def state_input_options(f):
    """Create common options for I/O files"""
    f = click.option('--in_file', '--in', '-i', type=click.Path('r'),
                     help='The state file that includes a dictionary-like specifications of descriptors to use.')(f)
    return f


def file_input_options(f):
    """Create common options for I/O files"""
    f = click.option('--fxyz', '-f',
                     type=str,
                     help='Input file that contains XYZ coordinates.\
                           See a list of possible input formats:\
                          https://wiki.fysik.dtu.dk/ase/ase/io/io.html\
                          If a wildcard * is used, all files matching the pattern is read.',
                     default=None)(f)
    return f


def file_input_format_options(f):
    """Create common options for I/O files"""
    f = click.option('--fxyz_format',
                     type=str,
                     help='Additional info for the input file format. e.g.\
                          {"format":"lammps-data","units":"metal","style":"full"}',
                     default=None)(f)
    return f


def file_output_options(f):
    """Create common options for I/O files"""
    f = click.option('--prefix', '-p',
                     help='Prefix to be used for the output file.',
                     default=None)(f)
    return f


def dm_input_options(f):
    """common options for reading a design matrices, used for map, fit, kde, clustering, etc."""
    f = click.option('--design_matrix', '-dm', cls=ConvertStrToList, default='[]',
                     help='Location of descriptor matrix file or name of the tags in ase xyz file\
                           the type is a list  \'[dm1, dm2]\', as we can put together simutanously several design matrix.')(
        f)
    f = click.option('--use_atomic_descriptors', '--use_atomic', '-ua',
                     help='Use atomic descriptors instead of global ones.',
                     default=False, is_flag=True)(f)
    f = click.option('--only_use_species', type=int,
                     help='Only use the atomic descriptors of species with the specified atomic number.\
                           Only makes sense if already using --use_atomic_descriptors.',
                     default=None)(f)
    return f


def km_input_options(f):
    """common options for reading a kernel matrices, can be used for map, fit, kde, clustering, etc."""
    f = click.option('--kernel_matrix', '-km', default='none',
                     help='Location of a kernel matrix file')(f)
    return f


def output_setup_options(f):
    """Create common options for output results from clustering/KDE analysis"""
    f = click.option('--savexyz/--no-savexyz',
                     help='Save the results to the xyz file',
                     default=True)(f)
    f = click.option('--savetxt/--no-savetxt',
                     help='Save the results to the txt file',
                     default=False)(f)
    return f


def desc_options(f):
    """Create common options for computing descriptors"""
    f = click.option('--tag',
                     help='Tag for the descriptors.',
                     default='cmd-desc')(f)
    return f


def para_options(f):
    """Create common options for parallellization"""
    f = click.option('--number_processes', '--nprocess', '-np', type=int,
                     help='Number of processes when compute the descriptors in parrallel.',
                     show_default=True, default=1)(f)
    return f


def atomic_to_global_desc_options(f):
    """Create common options for global descriptors constructed based on atomic fingerprints """
    f = click.option('--reducer_type', '-r',
                     help='type of operations to get global descriptors from the atomic soap vectors, e.g. \
                          [average], [sum], [moment_avg], [moment_sum].',
                     show_default=True, default='average', type=str)(f)
    f = click.option('--zeta', '-z',
                     help='Moments to take when converting atomic descriptors to global ones.',
                     default=1, type=int)(f)
    f = click.option('--element_wise', '-e',
                     help='element-wise operation to get global descriptors from the atomic soap vectors',
                     show_default=True, default=False, is_flag=True)(f)
    f = click.option('--peratom', '-pa',
                     help='Save the per-atom local descriptors.',
                     show_default=True, default=False, is_flag=True)(f)
    return f


def map_setup_options(f):
    """Create common options for making 2D maps of the data set"""
    f = click.option('--peratom',
                     help='Save the per-atom projection.',
                     default=False, is_flag=True)(f)
    f = click.option('--adjusttext/--no-adjusttext',
                     help='Adjust the annotation texts so they do not overlap.',
                     default=False)(f)
    f = click.option('--annotate', '-a',
                     help='Location of tags to annotate the samples.',
                     default='none', type=str)(f)
    f = click.option('--aspect_ratio', '-ar',
                     help='Aspect ratio of the plot',
                     show_default=True, default=2, type=float)(f)
    f = click.option('--style', '-s',
                     type=click.Choice(['default', 'journal'], case_sensitive=False),
                     help='Style of the plot.',
                     show_default=True, default='default')(f)
    return f


def map_io_options(f):
    """Create common options for making 2D maps of the data set"""
    f = click.option('--keepraw/--no-keepraw',
                     help='Keep the high dimensional descriptor when output XYZ file.',
                     default=False)(f)
    f = click.option('--output', '-o', type=click.Choice(['xyz', 'matrix', 'none', 'chemiscope'], case_sensitive=False),
                     help='Output file format.',
                     default='xyz')(f)
    f = click.option('--extra-properties', '-ep', type=click.Path(exists=True),
                     help='Additional properties to be read for each frmae in CSV format.')(f)
    return f


def color_setup_options(f):
    """Create common options for handing color scales"""
    f = click.option('--normalized_by_size', '-nbs',
                     help='Normalize the quantity used for color function by the number of atoms in each frame.',
                     show_default=True, default=False, is_flag=True)(f)
    f = click.option('--colormap', '-cmap',
                     help='Colormap used. Common options: gnuplot, tab10, viridis, bwr, rainbow.',
                     show_default=True, default='gnuplot')(f)
    f = click.option('--color_from_zero', '-c0',
                     help='Set the minimum to zero and only plot the excess.',
                     show_default=True, default=False, is_flag=True)(f)
    f = click.option('--color_label', '-clab',
                     help='The label for the color bar.',
                     default=None)(f)
    f = click.option('--color_column', '-ccol',
                     help='The column number used in the color file. Starts from 0.',
                     default=0)(f)
    f = click.option('--color', '-c',
                     help='Location of a file or name of the properties in the XYZ file. \
                     Used to color the scatter plot for all samples (N floats).',
                     default='none', type=str)(f)
    return f


def d_reduce_options(f):
    """Create common options for dimensionality reduction"""
    f = click.option('--axes', nargs=2, type=click.Tuple([int, int]),
                     help='Plot the projection along which projection axes.',
                     default=[0, 1])(f)
    f = click.option('--dimension', '-d',
                     help='Number of the dimensions to keep in the output XYZ file.',
                     default=10)(f)
    f = click.option('--scale/--no-scale',
                     help='Standard scaling of the coordinates.',
                     default=True)(f)
    return f


def fit_setup_options(f):
    """Create common options for making 2D maps of the data set"""
    f = click.option('--lc_points', '-lcp', type=int,
                     help='the number of sub-samples to take when compute the learning curve',
                     show_default=True, default=8)(f)
    f = click.option('--learning_curve', '-lc', type=int,
                     help='the number of points on the learning curve, <= 1 means no learning curve',
                     show_default=True, default=-1)(f)
    f = click.option('--test_ratio', '--test', '-t', type=float,
                     help='Test ratio.',
                     show_default=True, default=0.05)(f)
    f = click.option('--normalized_by_size', '-nbs',
                     help='Normalize y by the number of atoms in each frame.',
                     show_default=True, default=False, is_flag=True)(f)
    f = click.option('--y', '-y',
                     help='Location of a file or name of the properties in the XYZ file',
                     default='none', type=str)(f)
    return f


def kernel_options(f):
    """Create common options for compute kernel functions"""
    f = click.option('--kernel_parameter', '-kp', type=float,
                     help='Parameter used in the kernel function.',
                     default=None)(f)
    f = click.option('--kernel', '-k',
                     type=click.Choice(['linear', 'polynomial', 'cosine'], case_sensitive=False),
                     help='Kernel function for converting design matrix to kernel matrix.',
                     show_default=True, default='linear')(f)
    return f


def sparsification_options(f):
    """Create common options for sparsification"""
    f = click.option('--n_sparse', '-n', type=int,
                     help='number of the representative samples, set negative if using no sparsification',
                     show_default=True, default=100)(f)
    f = click.option('--sparse_mode', '-s',
                     type=click.Choice(['random', 'cur', 'fps', 'sequential'], case_sensitive=False),
                     help='Sparsification method to use.',
                     show_default=True, default='fps')(f)
    return f
