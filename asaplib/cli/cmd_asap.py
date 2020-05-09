"""
Module containing the top level asap command
"""
import json
from yaml import load as yload
from yaml import Loader
import numpy as np
import click

from asaplib.data import ASAPXYZ
from asaplib.hypers import universal_soap_hyper


@click.group('asap')
@click.pass_context
def asap(ctx):
    # Configure, make sure we have a dict object
    ctx.ensure_object(dict)


def desc_options(f):
    """Create common options for a descriptor command"""
    f = click.option('--fxyz', type=click.Path('r'), help='Input XYZ file')(f)
    f = click.option('--tag',
                     help='Tag for the descriptor output',
                     default='ASAP')(f)
    f = click.option('--prefix', help='Prefix to be used', default='ASAP')(f)
    f = click.option('--periodic/--no-periodic', 
                     help='Is the system periodic? If not specified, will infer from the XYZ file.',
                     default=True)(f)
    f = click.option('--stride',
                     help='Read in the xyz trajectory with X stide. Default: read/compute all frames',
                     default=1)(f)
    f = click.option('--state-file', type=click.Path('r'),
                     help='The state file that includes a dictionary-like specifications of descriptors to use.')(f)
    return f

def atomic_to_global_desc_options(f):
    """Create common options for global descriptors constructed based on atomic fingerprints """
    f = click.option('--kernel_type',
                     help='type of operations to get global descriptors from the atomic soap vectors [average], [sum], [moment_avg], [moment_sum]',
                     default='average')(f)
    f = click.option('--zeta', default=1, type=int)(f)
    f = click.option('--element-wise', default=False, is_flag=True)(f)
    f = click.option('--peratom', 
                     help='Save the per-atom local descriptors.',
                     default=False, is_flag=True)(f)
    return f

@asap.group('gen_desc')
@click.pass_context
@desc_options
@atomic_to_global_desc_options
def gen_desc(ctx, fxyz, tag, prefix, peratom, kernel_type, element_wise,
             stride, periodic, zeta, state_file):
    """
    Descriptor generation sub-command
    This command function evaluated before the descriptor specific ones,
    we setup the general stuff here, such as read the files.
    """

    if state_file:
        state = yload(state_file, Loader=Loader)
        # Here goes the routine to compute the descriptors according to the
        # state file(s)

    ctx.obj['kernel'] = {
        'kernel_type': kernel_type,
        'element_wise': element_wise,
    }
    if kernel_type == 'moment_average':
        ctx.obj['kernel']['zeta'] = zeta

    asapxyz = ASAPXYZ(fxyz, stride, periodic)

    ctx.obj['asapxyz'] = asapxyz
    ctx.obj['desc_settings'] = {
        'tag': tag,
        'prefix': prefix,
        'peratom': peratom,
    }

    """
    setting up the kernel function that is used to convert atomic descriptors into global descriptors for a structure.
    At the moment only one single kernel function can be used.
    """

@gen_desc.command('soap')
@click.option('--cutoff', type=float, help='Cutoff radius', default=3.0)
@click.option('--nmax', '-n', type=int, help='Maximum radial label', default=6)
@click.option('--lmax', '-l', type=int, help='Maximum angular label (<= 9)', default=6)
@click.option('--rbf', type=str, help='Radial basis function [gto] or [polynomial]', default='gto')
@click.option('--atom-gaussian-width', '-sigma', '-g', type=float, help='Atom width', default=0.5)
@click.option('--crossover/--no-crossover', help='If to included the crossover of atomic types.', default=False)
@click.pass_context
def soap(ctx, cutoff, nmax, lmax, atom_gaussian_width, crossover, rbf):
    """Generate SOAP descriptors"""

    # Read frames
    asapxyz = ctx.obj['asapxyz']
    desc_settings = ctx.obj['desc_settings']

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
    # The sepcification for the kernels
    kernel_spec = dict(k1=ctx.obj['kernel'])
    # The sepcification for the descriptor
    desc_spec = {
        'soap': {
            'atomic_descriptor': soap_spec,
            'kernel_function': kernel_spec,
        }
    }
    # Compute the save the descriptors
    output_desc(asapxyz, desc_spec, desc_settings)


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
