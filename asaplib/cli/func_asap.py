"""
functions used in cmd_asap.py
"""

import os
import json
from yaml import full_load as yload
import numpy as np
from matplotlib import pyplot as plt

from asaplib.data import ASAPXYZ, Design_Matrix
from asaplib.reducedim import Dimension_Reducers
from asaplib.plot import Plotters, set_color_function
from asaplib.io import ConvertStrToList

"""for loading in_file"""
def load_in_file(in_file):
    """Here goes the routine to compute the descriptors according to the state file(s)"""
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
    return state

"""for gen_desc"""
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

def output_desc(asapxyz, desc_spec, prefix, peratom=False):
    """
    Compute and save the descriptors
    """
    for tag, desc in desc_spec.items():
        # compute the descripitors
        asapxyz.compute_global_descriptors(desc_spec_dict=desc,
                                       sbs=[],
                                       keep_atomic=peratom,
                                       tag=tag)
    asapxyz.write(prefix)
    asapxyz.save_state(prefix)

""" for maps and fits """
def read_xyz_n_dm(fxyz, design_matrix, use_atomic_descriptors, peratom):
    dm = []; dm_atomic = []
    # try to read the xyz file
    if fxyz != 'none':
        asapxyz = ASAPXYZ(fxyz)
        if use_atomic_descriptors:
            _, dm = asapxyz.get_descriptors(design_matrix, True)
        else:
            dm, dm_atomic = asapxyz.get_descriptors(design_matrix, peratom)
    else:
        asapxyz = None
        print("Did not provide the xyz file. We can only output descriptor matrix.")
    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(design_matrix[0]):
        try:
            dm = np.genfromtxt(design_matrix[0], dtype=float)
            print("loaded the descriptor matrix from file: ", design_matrix[0])
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    return asapxyz, dm, dm_atomic

"""for maps"""
def map_process(obj, reduce_dict, axes, map_name):
    """
    process the dimensionality reduction command
    """
    # project
    if 'type' in reduce_dict.keys() and reduce_dict['type'] == 'RAW':
        proj = obj['design_matrix']
        if obj['map_info']['peratom']: 
            proj_atomic = obj['design_matrix_atomic']
        else:
            proj_atomic = None
    else:
        dreducer = Dimension_Reducers(reduce_dict)
        proj = dreducer.fit_transform(obj['design_matrix'])
        if obj['map_info']['peratom']:
            proj_atomic = dreducer.transform(obj['design_matrix_atomic'])
        else:
            proj_atomic = None
    # plot
    fig_spec = obj['fig_spec_dict']
    plotcolor = obj['map_info']['color']
    plotcolor_atomic = obj['map_info']['color_atomic']
    annotate = obj['map_info']['annotate']
    map_plot(fig_spec, proj, proj_atomic, plotcolor, plotcolor_atomic, annotate, axes)
    # output 
    outfilename = obj['fig_spec_dict']['outfile']
    outmode = obj['map_info']['outmode']
    if obj['map_info']['project_atomic']:
        map_save(outfilename, outmode, obj['asapxyz'], None, proj, map_name)
    else:
        map_save(outfilename, outmode, obj['asapxyz'], proj, proj_atomic, map_name)

def map_plot(fig_spec, proj, proj_atomic, plotcolor, plotcolor_atomic, annotate, axes):
    """
    Make plots
    """
    asap_plot = Plotters(fig_spec)
    asap_plot.plot(proj[::-1, axes], plotcolor[::-1], [], annotate)
    if proj_atomic is not None:
        asap_plot.plot(proj_atomic[::-1, axes], plotcolor_atomic[::-1],[],[])
    plt.show()

def map_save(foutput, outmode, asapxyz, proj, proj_atomic, map_name):
    """
    Save the low-D projections
    """
    if outmode == 'matrix':
        if proj is not None:
            np.savetxt(foutput + ".coord", proj, fmt='%4.8f', header='low D coordinates of samples')
        if proj_atomic is not None:
            np.savetxt(foutput + "-atomic.coord", proj_atomic, fmt='%4.8f', header=map_name)
    elif outmode == 'xyz':
        if proj is not None:
            asapxyz.set_descriptors(proj, map_name)
        if proj_atomic is not None:
            asapxyz.set_atomic_descriptors(proj_atomic, map_name)
        asapxyz.write(foutput)
    else:
        pass

""" for clustering """
def cluster_process(asapxyz, trainer, design_matrix, cluster_options):
    """handle clustering operations"""
    prefix = cluster_options['prefix']

    from asaplib.cluster import DBCluster
    do_clustering = DBCluster(trainer)
    do_clustering.fit(design_matrix)

    do_clustering.save_state(prefix)

    labels_db = do_clustering.get_cluster_labels()
    if cluster_options['savexyz']:
        if asapxyz is not None and cluster_options['use_atomic_descriptors']:
            asapxyz.set_atomic_descriptors(labels_db, prefix+'_cluster_label')
        elif asapxyz is not None:
            asapxyz.set_descriptors(labels_db, prefix+'_cluster_label')
        asapxyz.write(prefix)
    if cluster_options['savetxt']: 
        np.savetxt(prefix + "-cluster-label.dat", np.transpose([np.arange(len(labels_db)), labels_db]),
               header='index cluster_label', fmt='%d %d')

    # TODO: allow plotting options!

""" for KDE """
def kde_process(asapxyz, density_model, proj, kde_options):
    """handle kernel density estimation operations"""
    # fit density model to data
    try:
        density_model.fit(proj)
    except:
        raise RuntimeError('KDE did not work. Try smaller dimension may help.')

    rho = density_model.evaluate_density(proj)
    prefix = kde_options['prefix']
    # save the density
    if kde_options['savetxt']:
        np.savetxt(prefix + "-kde.dat", np.transpose([np.arange(len(rho)), rho]),
               header='index log_of_kernel_density_estimation', fmt='%d %4.8f')
    if kde_options['savexyz']:
        if asapxyz is not None and kde_options['use_atomic_descriptors']:
            asapxyz.set_atomic_descriptors(rho, density_model.get_acronym())
        elif asapxyz is not None:
            asapxyz.set_descriptors(rho, density_model.get_acronym())
        asapxyz.write(prefix)
