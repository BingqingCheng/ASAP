"""
Methods and functions to compute global descriptors of a frame from its atomic desciptors
"""

import numpy as np
import json
from ..io import NpEncoder, list2str

class Atomic_2_Global_Descriptors:
    def __init__(self, k_spec_dict):
        """
        Object handing the kernel functions used to convert atomic descriptors into global ones

        Parameters
        ----------
        k_spec_dict: dictionaries that specify which atomic to global descriptor to use 
        e.g.
        k_spec_dict = {'first_kernel': {'kernel_type': kernel_type,  
                          'zeta_list': zeta_list,
                          'species': species,
                          'element_wise': element_wise}}
        """
        self.k_spec_dict = k_spec_dict
        # list of Atomic_2_Global_Descriptor objections
        self.engines = []
        self.acronym = ""

        self.bind()

    def add(self, k_spec, tag):
        """
        adding the specifications of a new Atomic_2_Global_Descriptor
        Parameters
        ----------
        k_spec: a dictionary that specify which Atomic_2_Global_Descriptor descriptor to use 
        """
        self.k_spec_dict[tag] = k_spec

    def pack(self):
        return json.dumps(self.k_spec_dict, sort_keys=True, cls=NpEncoder)

    def get_acronym(self):
        if self.acronym == "":
            for engine in self.engines: 
                self.acronym += engine.get_acronym()
        return self.acronym

    def bind(self):
        """
        binds the objects that actually compute the descriptors
        these objects need to have .create(atomic_desc) method to compute the global descriptors from atomic ones
        """
        # clear up the objects
        self.engines = []
        for element in self.k_spec_dict.keys():
            self.engines.append(self._call(self.k_spec_dict[element]))
 
    def _call(self, k_spec):
        """
        call the specific descriptor objects
        """
        if "kernel_type" not in k_spec.keys():
            raise ValueError("Did not specify the type of the global descriptor kernel.")
        if k_spec["kernel_type"] == "average":
            return Atomic_2_Global_Average(k_spec)
        if k_spec["kernel_type"] == "sum":
            return Atomic_2_Global_Sum(k_spec)
        if k_spec["kernel_type"] == "moment_average":
            return Atomic_2_Global_MomentAverage(k_spec)
        if k_spec["kernel_type"] == "moment_sum":
            return Atomic_2_Global_MomentSum(k_spec)
        else:
            raise NotImplementedError 

    def compute(self, atomic_desc):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        atomic_desc: np.matrix. [N_atoms, N_atomi_desc]. Atomic descriptors for a frame.

        Returns
        -------
        desc: np.array [N_desc]. Global descriptors for a frame.
        """
        desc = []
        for engine in self.engines:
            desc = np.append(desc, engine.create(atomic_desc), axis=0)
        return desc

class Atomic_2_Global_Base:
    def __init__(self, k_spec):
        self.acronym = ""
        # we have defaults here; the default is not to distinguish between different elements
        if 'element_wise' in k_spec.keys():
            self.element_wise = bool(k_spec['element_wise'])
        else:
            self.element_wise = False

        if self.element_wise:
            try:
                self.species = k_spec['species']
            except:
                raise ValueError("Cannot do element-wise operations without specifying the global species")
            self.acronym = "e"

    def get_acronym(self):
        # we use an acronym for each descriptor, so it's easy to find it and refer to it
        return self.acronym

    def create(self, atomic_desc, atomic_numbers=[]):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.
        atomic_numbers: np.matrix. [N_atoms]. Atomic numbers for atoms in the frame.

        Returns
        -------
        desc: np.array [N_desc*len(global_species)]. Global descriptors for a frame.
        """
        pass


class Atomic_2_Global_Average(Atomic_2_Global_Base):
    """this is the vanilla situation. We just take the average soap for all atoms"""
    def __init__(self, k_spec):

        super().__init__(k_spec)
 
        if "kernel_type" not in k_spec.keys() or k_spec["kernel_type"] != "average":
            raise ValueError("kernel type is not average or cannot find the type")

    def create(self, atomic_desc, atomic_numbers=[]):
        if self.element_wise:
            return Descriptor_By_Species(atomic_desc, atomic_numbers, self.species, True)
        else:
            return np.mean(atomic_desc, axis=0)


class Atomic_2_Global_Sum(Atomic_2_Global_Base):
    """ We just take the sum soap for all atoms"""
    def __init__(self, k_spec):
        
        super().__init__(k_spec)

        if "kernel_type" not in k_spec.keys() or k_spec["kernel_type"] != "sum":
            raise ValueError("kernel type is not sum or cannot find the type")

        self.acronym += "-sum"

    def create(self, atomic_desc, atomic_numbers=[]):
        if self.element_wise:
            return Descriptor_By_Species(atomic_desc, atomic_numbers, self.species, False)
        else:
            return np.sum(atomic_desc, axis=0)

class Atomic_2_Global_Moment_Average(Atomic_2_Global_Base):
    """ 
    get the global descriptor from atomic ones 
    by averaging over the atomic descriptors of z th power 

    Parameters
    ----------
    zeta_list: moments considered
    """
    def __init__(self, k_spec_dict):


        super().__init__(k_spec)

        if "kernel_type" not in k_spec.keys() or k_spec["kernel_type"] != "moment_average":
            raise ValueError("kernel type is not moment_average or cannot find the type")

        try:
            self.zeta_list = k_spec['zeta_list']
        except:
            raise ValueError("cannot initialize the zeta list")

        self.acronym += "-z-"+list2str(zeta_list)

    def create(self, atomic_desc, atomic_numbers=[]):
        zeta_desc = Get_Moment(atomic_desc, self.zeta_list)
        if self.element_wise:
            return Descriptor_By_Species(zeta_desc, atomic_numbers, self.species, True)
        else:
            return np.mean(zeta_desc, axis=0)

class Atomic_2_Global_Moment_Sum(Atomic_2_Global_Base):
    """ 
    get the global descriptor from atomic ones 
    by averaging over the atomic descriptors of z th power 

    Parameters
    ----------
    zeta_list: moments considered
    """
    def __init__(self, k_spec_dict):


        super().__init__(k_spec)

        if "kernel_type" not in k_spec.keys() or k_spec["kernel_type"] != "moment_sum":
            raise ValueError("kernel type is not moment_sum or cannot find the type")

        try:
            self.zeta_list = k_spec['zeta_list']
        except:
            raise ValueError("cannot initialize the zeta list")

        self.acronym += "-z-"+list2str(zeta_list)+"-sum"

    def create(self, atomic_desc, atomic_numbers=[]):
        zeta_desc = Get_Moment(atomic_desc, self.zeta_list)
        if self.element_wise:
            return Descriptor_By_Species(zeta_desc, atomic_numbers, self.species, False)
        else:
            return np.sum(zeta_desc, axis=0)

def Get_Moment(atomic_desc, zeta_list):
    """ 
    get the higher moments for atomic descriptors

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.
    zeta_list: moments considered
    """
    n_zeta = len(zeta_list)
    n_desc = len(desc_list)
    new_desc = np.zeros(n_zeta*n_desc)
    for i, z in enumerate(zeta_list):
        new_desc[i::n_zeta] = np.power(atomic_desc,z)
    return new_desc

def Descriptor_By_Species(atomic_desc, atomic_numbers, global_species, average_over_natom=True):
    """ 
    first compute the average/sum descriptors for each species,
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
    desc_by_species = {}
    desc_len = 0
    for species in global_species:
        atomicdesc_by_species = [atomic_desc[i] for i,at in enumerate(atomic_numbers) if at==species]
        desc_by_species[species] = np.sum(atomicdesc_by_species, axis=0)
        desc_len =  max(len(desc_by_species[species]), desc_len)

    desc = np.zeros(desc_len*len(global_species),dtype=float)
    for i, species in enumerate(global_species):
        if len(desc_by_species[species]) > 0:
            # normalize by the number of atoms
            if average_over_natom:
                desc_by_species[species] /=  len(desc_by_species[species])
            desc[i*desc_len:(i+1)*desc_len] = desc_by_species[species]
    return np.asarray(desc)
