"""
Methods and functions to obtain global descriptors from atomic descriptors
or the similarity/kernel between two structures using their atomic descriptors
"""

import numpy as np

def Atomic_2_Global_Descriptor_By_Species(atomic_desc, atomic_numbers=[], global_species=[],
                                      kernel_type='average', zeta_list = [1]):
    """ 
    first compute the average descriptors for each species,
    then concatenate them.

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.
    atomic_numbers: np.matrix. [N_atoms]. Atomic numbers for atoms in the frame.
    global_species: a list of all atomic species in all frames
    average_over_natom: normalized by number of the atoms of the same species

    Returns
    -------
    desc: np.array [N_desc*len(global_species)]. Global descriptors for a frame.
    """
    # don't distinguish species
    if len(global_species) == 0:
        return Atomic_2_Global_Descriptor(atomic_desc, kernel_type, zeta_list)

    desc_by_species = {}
    desc_len = 0
    for species in global_species:
        atomicdesc_by_species = [atomic_desc[i] for i,at in enumerate(atomic_numbers) if at==species]
        #print(species, np.shape(atomicdesc_by_species))
        if len(atomicdesc_by_species) > 0:
            desc_by_species[species] = Atomic_2_Global_Descriptor(atomicdesc_by_species, kernel_type, zeta_list)
        else:
            desc_by_species[species] = []
        desc_len =  max(len(desc_by_species[species]), desc_len)
        #print(species, np.shape(desc_by_species[species]))

    desc = np.zeros(desc_len*len(global_species),dtype=float)
    for i, species in enumerate(global_species):
        if len(desc_by_species[species]) > 0:
            desc[i*desc_len:(i+1)*desc_len] = desc_by_species[species]
    return np.asarray(desc)

def Atomic_2_Global_Descriptor(atomic_desc, kernel_type='average', zeta_list = [1]):
    """ Get global descriptor from atomic descriptor of the same species"""
    if kernel_type == 'average':
        if zeta_list==1 or ( len(zeta_list) == 1 and zeta_list[0]) == 1 :
            # average over all atomic environments inside the system
            return Avgerage_Descriptor(atomic_desc)
        else:
            # average of the moments of all atomic environments inside the system
            return Average_Moment_Descriptor(atomic_desc, zeta_list)

    elif kernel_type == 'sum':
        if zeta_list==1 or ( len(zeta_list) == 1 and zeta_list[0]) == 1 :
            # sum over all atomic environments inside the system
            return Sum_Descriptor(atomic_desc)
        else:
            # sum of the moments of all atomic environments inside the system
            return Sum_Moment_Descriptor(atomic_desc, zeta_list)
    else:
        raise NotImplementedError

def Avgerage_Descriptor(atomic_desc):
    """ get the global descriptor from atomic ones by a simple averaging

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.

    Returns
    -------
    desc: np.matrix [N_desc]. Global descriptors for a frame.
    """
    return np.mean(atomic_desc, axis=0)

def Sum_Descriptor(atomic_desc):
    """ get the global descriptor from atomic ones by summing over atomic ones

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.

    Returns
    -------
    desc: np.matrix [N_desc]. Global descriptors for a frame.
    """
    return np.sum(atomic_desc,axis=0)

def Descriptor_By_Species_old(atomic_desc, atomic_numbers, global_species, average_over_natom=True):
    """ 
    first compute the average descriptors for each species,
    then concatenate them.

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.
    atomic_numbers: np.matrix. [N_atoms]. Atomic numbers for atoms in the frame.
    global_species: a list of all atomic species in all frames
    average_over_natom: normalized by number of the atoms of the same species

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
        desc_by_species[atomic_numbers[at]][:] += at_desc[:]
        natoms_by_species[species] += 1

    for i, species in enumerate(global_species):
        # normalize by the number of atoms
        if average_over_natom:
            if natoms_by_species[species] > 0:
                desc_by_species[species] /=  natoms_by_species[species]
        desc[i*n_adesc:(i+1)*n_adesc] = desc_by_species[species]
    return desc

def Average_Moment_Descriptor(atomic_desc, zeta_list=[1,2,3]):
    """ get the global descriptor from atomic ones 
        by averaging over the atomic descriptors of z th power 

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.
    zeta: highest moment considered
    Returns
    -------
    desc: np.matrix [N_desc*zeta] or [N_desc*zeta+1]. Global descriptors for a frame.
    """
    n_adesc = len(atomic_desc[0])
    n_moment = len(zeta_list)
    desc = np.zeros(n_adesc*n_moment,dtype=float)

    desc[:n_adesc] = np.mean(atomic_desc, axis=0)
    for i, z in enumerate(zeta_list):
        desc[i*n_adesc:(i+1)*n_adesc] = np.mean(np.power(atomic_desc,z), axis=0)
    return desc

def Sum_Moment_Descriptor(atomic_desc, zeta_list=[0,1,2,3]):
    """ get the global descriptor from atomic ones 
        by summing over the atomic descriptors of z th power

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.
    zeta: highest moment considered
    Returns
    -------
    desc: np.matrix [N_desc*zeta] or [N_desc*zeta+1]. Global descriptors for a frame.
    """
    n_adesc = len(atomic_desc[0])
    n_moment = len(zeta_list)
    desc = np.zeros(n_adesc*n_moment,dtype=float)

    desc[:n_adesc] = np.mean(atomic_desc, axis=0)
    for i, z in enumerate(zeta_list):
        desc[i*n_adesc:(i+1)*n_adesc] = np.sum(np.power(atomic_desc,z), axis=0)
    return desc

