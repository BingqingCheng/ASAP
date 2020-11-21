"""
Adaptors for generating ChemiScope compatible inputs
"""

import warnings
import numpy as np
import json
import gzip

IGNORED_ASE_ARRAYS = ['positions', 'numbers', 'stress']


def _typetransform(data):
    """Ensure data is a list of JSON serialisable objects"""
    assert isinstance(data, list) and len(data) > 0
    if isinstance(data[0], str):
        return list(map(str, data))
    elif isinstance(data[0], bytes):
        return list(map(lambda u: u.decode('utf8'), data))
    else:
        try:
            if isinstance(data[0], float):
                return [float(value) for value in data]
            else:
                # 2D
                return [float(np.asarray(value).ravel()[0]) for value in data]
        except ValueError:
            raise Exception('unsupported type in value')


def _linearize(name, value):
    """
    Transform 2D arrays in multiple 1D arrays, converting types to fit json as
    needed.
    """
    data = {}
    if isinstance(value['values'], list):
        data[name] = {
            'target': value['target'],
            'values': _typetransform(value['values']),
        }
    elif isinstance(value['values'], np.ndarray):
        if len(value['values'].shape) == 1:
            data[name] = {
                'target': value['target'],
                'values': _typetransform(list(value['values'])),
            }
        elif len(value['values'].shape) == 2:
            for i in range(value['values'].shape[1]):
                data[f'{name}[{i + 1}]'] = {
                    'target': value['target'],
                    'values': _typetransform(list(value['values'][:, i])),
                }
        else:
            raise Exception('unsupported ndarray value')
    else:
        raise Exception(f'unknown type for value {name}')

    return data


def _frame_to_json(frame):
    data = {}
    data['size'] = len(frame)
    data['names'] = list(frame.symbols)
    data['x'] = [float(value) for value in frame.positions[:, 0]]
    data['y'] = [float(value) for value in frame.positions[:, 1]]
    data['z'] = [float(value) for value in frame.positions[:, 2]]

    if (frame.cell.lengths() != [0.0, 0.0, 0.0]).all():
        data['cell'] = list(np.concatenate(frame.cell))

    return data


def _generate_environments(frames, cutoff):
    environments = []
    for frame_id, frame in enumerate(frames):
        for center in range(len(frame)):
            environments.append({
                'structure': frame_id,
                'center': center,
                'cutoff': cutoff,
            })
    return environments


def write_chemiscope_input(filename,
                           frames,
                           meta=None,
                           extra=None,
                           cutoff=None):
    """
    Write the json file expected by the default chemiscope visualizer at
    ``filename``.
    :param str filename: name of the file to use to save the json data. If it
                         ends with '.gz', a gzip compressed file will be written
    :param list frames: list of `ase.Atoms`_ objects containing all the
                        structures
    :param dict meta: optional metadata of the dataset, see below
    :param dict extra: optional dictionary of additional properties, see below
    :param float cutoff: optional. If present, will be used to generate
                         atom-centered environments
    The dataset metadata should be given in the ``meta`` dictionary, the
    possible keys are:
    .. code-block:: python
        meta = {
            'name': '...',         # str, dataset name
            'description': '...',  # str, dataset description
            'authors': [           # list of str, dataset authors, OPTIONAL
                '...',
            ],
            'references': [        # list of str, references for this dataset,
                '...',             # OPTIONAL
            ],
        }
    The written JSON file will contain all the properties defined on the
    `ase.Atoms`_ objects. Values in ``ase.Atoms.arrays`` are mapped to
    ``target = "atom"`` properties; while values in ``ase.Atoms.info`` are
    mapped to ``target = "structure"`` properties. The only exception is
    ``ase.Atoms.arrays["numbers"]``, which is always ignored. If you want to
    have the atomic numbers as a property, you should add it to ``extra``
    manually.
    Additional properties can be added with the ``extra`` parameter. This
    parameter should be a dictionary containing one entry for each property.
    Each entry contains a ``target`` attribute (``'atom'`` or ``'structure'``)
    and a set of values. ``values`` can be a Python list of float or string; a
    1D numpy array of numeric values; or a 2D numpy array of numeric values. In
    the later case, multiple properties will be generated along the second axis.
    For example, passing
    .. code-block:: python
        extra = {
            'cheese': {
                'target': 'atom',
                'values': np.zeros((300, 4))
            }
        }
    will generate four properties named ``cheese[1]``, ``cheese[2]``,
    ``cheese[3]``,  and ``cheese[4]``, each containing 300 values.
    .. _`ase.Atoms`: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
    
    :NOTE:
      Adapted from: https://github.com/cosmo-epfl/chemiscope/blob/master/utils/chemiscope_input.py
    """

    if not (filename.endswith('.json') or filename.endswith('.json.gz')):
        raise Exception('filename should end with .json or .json.gz')

    data = {'meta': {}}

    if meta is not None:
        if 'name' in meta:
            data['meta']['name'] = str(meta['name'])

        if 'description' in meta:
            data['meta']['description'] = str(meta['description'])

        if 'authors' in meta:
            data['meta']['authors'] = list(map(str, meta['authors']))

        if 'references' in meta:
            data['meta']['references'] = list(map(str, meta['references']))

        for key in meta.keys():
            if key not in ['name', 'description', 'authors', 'references']:
                warnings.warn('ignoring unexpected metadata: {}'.format(key))

    if 'name' not in data['meta'] or not data['meta']['name']:
        data['meta']['name'] = filename

    properties = {}
    if extra is not None:
        for name, value in extra.items():
            properties.update(_linearize(name, value))

    # Read properties coming from the ase.Atoms objects
    from_frames = {}

    # target: structure properties
    # TODO this need to updates as ASAP store arrays for each atom in the INFO?
    def _append_value(from_frames, name, value):
        """Append value to from_frames, create the entry if not exists"""
        if name in from_frames:
            from_frames[name]['values'].append(value)
        else:
            from_frames.update(
                {name: {
                    'target': 'structure',
                    'values': [value]
                }})
        return

    def _extend_value(from_frames, name, value):
        """Extend the entry in from_frames, create the entry if not exists"""
        if name in from_frames:
            from_frames[name]['values'].extend(value)
        else:
            from_frames.update(
                {name: {
                    'target': 'atom',
                    'values': list(value)
                }})
        return


    for frame in frames:
        for name, value in frame.info.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for idx, _value in enumerate(value):
                    _name = name + f'[{idx}]'
                    _append_value(from_frames, _name, _value)
            else:
                _append_value(from_frames, name, value)

    # target: atom properties
    has_atomic = False
    for frame in frames:
        for name, value in frame.arrays.items():
            if name in IGNORED_ASE_ARRAYS:
                continue
            has_atomic = True
            if len(value.shape) > 1:
                # Iterate over the columns
                for idx, _value in enumerate(value.T):
                    _name = 'atomic-' + name + f'[{idx}]'
                    _extend_value(from_frames, _name, _value)
            else:
                _extend_value(from_frames, name, value)

    for name, value in from_frames.items():
        properties.update(_linearize(name, value))

    data['properties'] = properties
    data['structures'] = [_frame_to_json(frame) for frame in frames]

    if cutoff is not None and has_atomic:
        data['environments'] = _generate_environments(frames, cutoff)

    if filename.endswith(".gz"):
        with gzip.open(filename, 'w', 9) as file:
            file.write(json.dumps(data).encode("utf8"))
    else:
        with open(filename, 'w') as file:
            json.dump(data, file)
