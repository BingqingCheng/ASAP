"""
Functions for IO and type conversions
"""

import argparse
import json
import click
import ast

import numpy as np
import random
import string

def extract_from_nested_dict(key, var):
    if hasattr(var,'iteritems'):
        for k, v in var.iteritems():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in extract_from_nested_dict(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in extract_from_nested_dict(key, d):
                        yield result

def str2bool(v):
    """

    Parameters
    ----------
    v

    Returns
    -------

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def list2str(input_list):
    output_str = ""
    for l in input_list:
        output_str += str(l)
        output_str += "-"
    return output_str

class ConvertStrToList(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            value = str(value)
            assert value.count('[') == 1 and value.count(']') == 1
            list_as_str = value.replace('"', "'").split('[')[1].split(']')[0]
            list_of_items = [item.strip().strip("'") for item in list_as_str.split(',')]
            return list_of_items
        except Exception:
            raise click.BadParameter(value)

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class NpDecoder(json.JSONDecoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpDecoder, self).default(obj)
