"""
functions used in cmd_asap.py
"""

"""for loading in_file"""


def load_in_file(in_file):
    """Here goes the routine to compute the descriptors according to the state file(s)"""

    with open(in_file, 'r') as stream:
        try:
            from yaml import full_load as yload
            state_str = yload(stream)
        except:
            import json
            state_str = json.load(stream)
    state = {}
    for k, s in state_str.items():
        if isinstance(s, str):
            state[k] = json.loads(s)
        else:
            state[k] = s
    return state


""" for load ASAPXYZ """


def load_asapxyz(data_spec):
    from asaplib.data import ASAPXYZ
    return ASAPXYZ(data_spec['fxyz'], data_spec['stride'], data_spec['periodic'], data_spec['fxyz_format'])


"""for gen_desc"""


def set_reducer(reducer_type, element_wise, zeta):
    """
    setting up the reducer function that is used to convert atomic descriptors into global descriptors for a structure.
    At the moment only one single reducer function can be used.
    """
    reducer_func = {'reducer1': {
        'reducer_type': reducer_type,  # [average], [sum], [moment_average], [moment_sum]
        'element_wise': element_wise,
    }}
    if reducer_type == 'moment_average' or reducer_type == 'moment_sum':
        reducer_func['zeta'] = zeta
    return reducer_func


def output_desc(asapxyz, desc_spec, prefix, peratom=False, N_processes=8):
    """
    Compute and save the descriptors
    """
    for tag, desc in desc_spec.items():
        # compute the descripitors
        asapxyz.compute_global_descriptors(desc_spec_dict=desc,
                                           sbs=[],
                                           keep_atomic=peratom,
                                           tag=tag,
                                           n_process=N_processes)
    asapxyz.write(prefix)
    asapxyz.save_state(prefix)


""" for maps and fits """


def read_xyz_n_dm(fxyz, design_matrix, use_atomic_descriptors, only_use_species, peratom):
    dm = []
    dm_atomic = []
    # try to read the xyz file
    if fxyz is not None and fxyz != 'none':
        from asaplib.data import ASAPXYZ
        asapxyz = ASAPXYZ(fxyz)
        if use_atomic_descriptors:
            dm = asapxyz.get_atomic_descriptors(design_matrix, only_use_species)
        else:
            dm, dm_atomic = asapxyz.get_descriptors(design_matrix, peratom)
    else:
        asapxyz = None
        print("Did not provide the xyz file. We can only output descriptor matrix.")
    # we can also load the descriptor matrix from a standalone file
    import os
    if os.path.isfile(design_matrix[0]):
        try:
            import numpy as np
            dm = np.genfromtxt(design_matrix[0], dtype=float)
            print("loaded the descriptor matrix from file: ", design_matrix[0])
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    return asapxyz, dm, dm_atomic


"""for maps"""


def figure_style_setups(prefix,
                        colorlabel, colorscale, colormap,
                        style, aspect_ratio, adjusttext):
    fig_options = {'outfile': prefix,
                   'show': False,
                   'title': None,
                   'size': [8 * aspect_ratio, 8],
                   'cmap': colormap,
                   'components': {
                       'first_p': {'type': 'scatter', 'clabel': colorlabel,
                                   'vmin': colorscale[0], 'vmax': colorscale[1]},
                       'second_p': {"type": 'annotate', 'adtext': adjusttext}}
                   }
    if style == 'journal':
        fig_options.update({'xlabel': None, 'ylabel': None,
                            'xaxis': False, 'yaxis': False,
                            'remove_tick': True,
                            'rasterized': True,
                            'fontsize': 12,
                            'size': [4 * aspect_ratio, 4]
                            })
    return fig_options


def map_process(obj, reduce_dict, axes, map_name):
    """
    process the dimensionality reduction command
    """
    # project
    if 'type' in reduce_dict.keys() and reduce_dict['type'] == 'RAW':
        proj = obj['design_matrix']
        if obj['map_options']['peratom']:
            proj_atomic = obj['design_matrix_atomic']
        else:
            proj_atomic = None
    else:
        from asaplib.reducedim import Dimension_Reducers
        dreducer = Dimension_Reducers(reduce_dict)
        proj = dreducer.fit_transform(obj['design_matrix'])
        if obj['map_options']['peratom']:
            print("Project atomic design matrix with No. of samples:", len(obj['design_matrix_atomic']))
            proj_atomic = dreducer.transform(obj['design_matrix_atomic'])
        else:
            proj_atomic = None
    # plot
    fig_spec = obj['fig_options']
    plotcolor = obj['map_options']['color']
    plotcolor_atomic = obj['map_options']['color_atomic']
    annotate = obj['map_options']['annotate']
    if 'cluster_labels' in obj.keys():
        labels = obj['cluster_labels']
    else:
        labels = []
    map_plot(fig_spec, proj, proj_atomic, plotcolor, plotcolor_atomic, labels, annotate, axes)
    # output 
    outfilename = obj['fig_options']['outfile']
    outmode = obj['map_options']['outmode']
    species_name = obj['map_options']['only_use_species']
    if obj['map_options']['project_atomic']:
        map_save(outfilename, outmode, obj['asapxyz'], None, proj, map_name, species_name)
    else:
        map_save(outfilename, outmode, obj['asapxyz'], proj, proj_atomic, map_name, species_name)


def map_plot(fig_spec, proj, proj_atomic, plotcolor, plotcolor_atomic, labels, annotate, axes):
    """
    Make plots
    """
    from matplotlib import pyplot as plt
    from asaplib.plot import Plotters
    asap_plot = Plotters(fig_spec)
    asap_plot.plot(proj[:, axes], plotcolor[:], labels[:], annotate[:])
    if proj_atomic is not None:
        asap_plot.plot(proj_atomic[:, axes], plotcolor_atomic[:], [], [])
    plt.show()


def map_save(foutput, outmode, asapxyz, proj, proj_atomic, map_name, species_name):
    """
    Save the low-D projections
    """
    if outmode == 'matrix':
        import numpy as np
        if proj is not None:
            np.savetxt(foutput + ".coord", proj, fmt='%4.8f', header='low D coordinates of samples')
        if proj_atomic is not None:
            np.savetxt(foutput + "-atomic.coord", proj_atomic, fmt='%4.8f', header=map_name)
    elif outmode in ('xyz', 'chemiscope'):
        if proj is not None:
            asapxyz.set_descriptors(proj, map_name)
        if proj_atomic is not None:
            asapxyz.set_atomic_descriptors(proj_atomic, map_name, species_name)
        if outmode == 'xyz':
            asapxyz.write(foutput)
        else:
            # If we write atomic projection assume we want to show them
            cutoff = 3.5 if proj_atomic else None
            asapxyz.write_chemiscope(foutput, cutoff=cutoff)
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
    if cluster_options['savexyz'] and asapxyz is not None:
        if cluster_options['use_atomic_descriptors']:
            asapxyz.set_atomic_descriptors(labels_db, prefix + '_cluster_label', cluster_options['only_use_species'])
        else:
            asapxyz.set_descriptors(labels_db, prefix + '_cluster_label')
        asapxyz.write(prefix)
    if cluster_options['savetxt']:
        import numpy as np
        np.savetxt(prefix + "-cluster-label.dat", np.transpose([np.arange(len(labels_db)), labels_db]),
                   header='index cluster_label', fmt='%d %d')

    return labels_db


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
    if kde_options['savexyz'] and asapxyz is not None:
        if kde_options['use_atomic_descriptors']:
            asapxyz.set_atomic_descriptors(rho, density_model.get_acronym(), kde_options['only_use_species'])
        else:
            asapxyz.set_descriptors(rho, density_model.get_acronym())
        asapxyz.write(prefix)
    return rho
