"""
Module containing the top level asap command
"""

import numpy as np
from asaplib.data import ASAPXYZ
from asaplib.hypers import universal_soap_hyper

import click


@click.group('asap')
@click.pass_context
def asap(ctx):
    # Configure, make sure we have a dict object
    ctx.ensure_object(dict)


def desc_options(f):
    """Create common options for a descriptor command"""
    f = click.option('--fxyz', type=click.Path('r'), help='Input XYZ file')(f)
    f = click.option('--tag', help='Tag for the descriptor output', default='ASAP')(f)
    f = click.option('--prefix', help='Prefix to be used', default='ASAP')(f)
    f = click.option('--peratom', is_flag=True, default=False)(f)
    f = click.option('--kernel_type', default='moment_average')(f)
    f = click.option('--zeta', default=2, type=int)(f)
    f = click.option('--element-wise', default=False, is_flag=True)(f)
    f = click.option('--periodic/--no-periodic', default=False)(f)
    f = click.option('--stride', default=1)(f)
    return f


@asap.group('gen_desc')
@click.pass_context
@desc_options
def gen_desc(ctx, fxyz, tag, prefix, peratom, kernel_type, element_wise,
             stride, periodic, zeta):
    """Descriptor generation sub-command"""
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

    # read frames
    asapxyz = ctx.obj['asapxyz']
    desc_settings = ctx.obj['desc_settings']

    soap_js = {
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
    kernel_js = dict(k1=ctx.obj['kernel'])
    desc_spec_js = {
        'soap': {
            'atomic_descriptor': soap_js,
            'kernel_function': kernel_js,
        }
    }

    # compute the descripitors
    asapxyz.compute_global_descriptors(desc_spec_js, [], desc_settings['peratom'], desc_settings['tag'])
    asapxyz.write(desc_settings['prefix'])
    asapxyz.save_state(desc_settings['tag'])
    asapxyz.save_descriptor_state(desc_settings['tag'])