"""
Methods and functions to obtain global descriptors from atomic descriptors
or the similarity/kernel between two structures using their atomic descriptors
"""

import numpy as np

def Avgerage_Descriptor(atomic_desc):
    """ get the global descriptor from atomic ones by a simple averaging

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.

    Returns
    -------
    desc: np.matrix [N_desc]. Global descriptors for a frame.
    """
    return atomic_desc.mean(axis=0)

def Avgerage_Descriptor_By_Species(atomic_desc, atomic_numbers, global_species):
    """ get the global descriptor from atomic ones by a simple averaging

    first compute the average descriptors for each species,
    then concatenate them.

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.
    atomic_numbers: np.matrix. [N_atoms]. Atomic numbers for atoms in the frame.
    global_species: a list of all atomic species in all frames
    Returns
    -------
    desc: np.matrix [N_desc*len(global_species)]. Global descriptors for a frame.
    """
    n_adesc = len(atomic_desc[0])
    desc = np.zeros(n_adesc*len(global_species),dtype=float)
    desc_by_species = {}
    natoms_by_species = {}
    for species in global_species:
        desc_by_species[species] = np.zeros(n_adesc,dtype=float)
        natoms_by_species[species] = 0
    for at, at_desc in enumerate(atomic_desc):
        desc_by_species[atomic_numbers(at)][:] += at_desc[:]
        natoms_by_species[species] += 1
    for i, species in enumerate(global_species):
        desc_by_species[species] /=  natoms_by_species[species]
        desc[i*n_adesc:(i+1)*n_adesc] = desc_by_species[species]
    return desc

