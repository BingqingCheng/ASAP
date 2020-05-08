"""
Module containing the top level asap command
"""
import json

import numpy as np
import click

from asaplib.data import ASAPXYZ
from asaplib.hypers import universal_soap_hyper
from asaplib.io.io_parse import NpDecoder



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
    f = click.option('--peratom', is_flag=True, default=False)(f)
    f = click.option('--kernel_type', default='moment_average')(f)
    f = click.option('--zeta', default=2, type=int)(f)
    f = click.option('--element-wise', default=False, is_flag=True)(f)
    f = click.option('--periodic/--no-periodic', default=False)(f)
    f = click.option('--stride', default=1)(f)
    f = click.option('--state-file', type=click.Path('r'))(f)
    return f


@asap.group('gen_desc')
@click.pass_context
@desc_options
def gen_desc(ctx, fxyz, tag, prefix, peratom, kernel_type, element_wise,
             stride, periodic, zeta, state_file):
    """
    Descriptor generation sub-command
    This command function evaluated before the descriptor specific ones,
    we setup the geral stuff here, such as read the files and setting up
    the kernel function.
    At the moment only one single kernel function can be used.
    """

    if state_file:
        state = json.load(state_file, decoder=NpDecoder)
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


@gen_desc.command('soap')
@click.option('--cutoff', type=float, default=3.0)
@click.option('--nmax', '-n', type=int, default=6)
@click.option('--lmax', '-l', type=int, default=6)
@click.option('--rbf', default='gto')
@click.option('--atom-gaussian-width', '-sigma', type=float, default=0.5)
@click.option('--crossover/--no-crossover', default=False)
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
    asapxyz.save_descriptor_state(tag)