"""
Methods and functions to compute global descriptors of a frame from its atomic desciptors
"""

import json

import numpy as np

from ..io import NpEncoder


class Atomic_2_Global_Descriptors:
    def __init__(self, k_spec_dict):
        """
        Object handing the reducer functions used to convert atomic descriptors into global ones

        Parameters
        ----------
        k_spec_dict: dictionaries that specify which atomic to global descriptor to use 
        e.g.
        k_spec_dict = {'first_reducer': {'reducer_type': reducer_type,  
                          'zeta': zeta,
                          'species': species,
                          'element_wise': element_wise}}
        """
        self.k_spec_dict = k_spec_dict
        # list of Atomic_2_Global_Descriptor objections
        self.engines = {}

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

    def bind(self):
        """
        binds the objects that actually compute the descriptors
        these objects need to have .create(atomic_desc) method to compute the global descriptors from atomic ones
        """
        # clear up the objects
        self.engines = {}
        for element in self.k_spec_dict.keys():
            self.engines[element] = self._call(self.k_spec_dict[element])
            self.k_spec_dict[element]['acronym'] = self.engines[element].get_acronym()

    def _call(self, k_spec):
        """
        call the specific descriptor objects
        """
        if "reducer_type" not in k_spec.keys():
            raise ValueError("Did not specify the type of the global descriptor reducer.")
        if k_spec["reducer_type"] == "average":
            return Atomic_2_Global_Average(k_spec)
        if k_spec["reducer_type"] == "sum":
            return Atomic_2_Global_Sum(k_spec)
        if k_spec["reducer_type"] == "moment_average":
            return Atomic_2_Global_Moment_Average(k_spec)
        if k_spec["reducer_type"] == "moment_sum":
            return Atomic_2_Global_Moment_Sum(k_spec)
        else:
            raise NotImplementedError

    def compute(self, atomic_desc_dict, atomic_numbers):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        atomic_desc_dict : a dictionary. each entry contains the essential info of the descriptor (acronym) 
                          and a np.array [N_desc*N_atoms]. Global descriptors for a frame.
                     see Atomic_Descriptors.compute() in .atomic_descriptors.py
        atomic_numbers: np.matrix. [N_atoms]. Atomic numbers for atoms in the frame.

        Returns
        -------
        desc_dict: a dictionary. each entry contains the essential info of the descriptor, i.e. acronym 
                          and a np.array [N_desc]. Global descriptors for a frame.
                   e.g. {'d1':{ 'acronym': 'XXX', 'descriptors': `a np.array [N_desc]`}}
        """
        desc_dict = {}
        for atomic_desc_element in atomic_desc_dict.keys():
            atomic_desc_now = atomic_desc_dict[atomic_desc_element]['atomic_descriptors']
            desc_dict[atomic_desc_element] = {}
            for element in self.k_spec_dict.keys():
                desc_dict[atomic_desc_element][element] = {}
                k_acronym, desc_dict[atomic_desc_element][element]['descriptors'] = self.engines[element].create(
                    atomic_desc_now, atomic_numbers)
                # we use a combination of the acronym of the descriptor and of the reducer function
                desc_dict[atomic_desc_element][element]['acronym'] = atomic_desc_dict[atomic_desc_element][
                                                                         'acronym'] + k_acronym
        return desc_dict


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
            self.acronym = "-e"

    def get_acronym(self):
        # we use an acronym for each descriptor, so it's easy to find it and refer to it
        return self.acronym

    def create(self, atomic_desc, atomic_numbers=[]):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        atomic_desc: a np.array [N_desc*N_atoms]. Atomic descriptors for a frame.
        atomic_numbers: np.matrix. [N_atoms]. Atomic numbers for atoms in the frame.

        Returns
        -------
        acronym: self.acronym
        desc:  a np.array [N_desc]. Global descriptors for a frame.
        """
        return self.acronym, []


class Atomic_2_Global_Average(Atomic_2_Global_Base):
    """this is the vanilla situation. We just take the average soap for all atoms"""

    def __init__(self, k_spec):

        super().__init__(k_spec)

        if "reducer_type" not in k_spec.keys() or k_spec["reducer_type"] != "average":
            raise ValueError("reducer type is not average or cannot find the type")

        print("Using Atomic_2_Global_Average reducer ...")

    def create(self, atomic_desc, atomic_numbers=[]):
        if self.element_wise:
            return self.acronym, Descriptor_By_Species(atomic_desc, atomic_numbers, self.species, True)
        else:
            return self.acronym, np.mean(atomic_desc, axis=0)


class Atomic_2_Global_Sum(Atomic_2_Global_Base):
    """ We just take the sum soap for all atoms"""

    def __init__(self, k_spec):

        super().__init__(k_spec)

        if "reducer_type" not in k_spec.keys() or k_spec["reducer_type"] != "sum":
            raise ValueError("reducer type is not sum or cannot find the type")

        print("Using Atomic_2_Global_Sum reducer ...")
        self.acronym += "-sum"

    def create(self, atomic_desc, atomic_numbers=[]):
        if self.element_wise:
            return self.acronym, Descriptor_By_Species(atomic_desc, atomic_numbers, self.species, False)
        else:
            return self.acronym, np.sum(atomic_desc, axis=0)


class Atomic_2_Global_Moment_Average(Atomic_2_Global_Base):
    """ 
    get the global descriptor from atomic ones 
    by averaging over the atomic descriptors of z th power 

    Parameters
    ----------
    zeta: take the zeta th power
    """

    def __init__(self, k_spec):

        super().__init__(k_spec)

        if "reducer_type" not in k_spec.keys() or k_spec["reducer_type"] != "moment_average":
            raise ValueError("reducer type is not moment_average or cannot find the type")

        try:
            self.zeta = k_spec['zeta']
        except:
            raise ValueError("cannot initialize the zeta value")

        print("Using Atomic_2_Global_Moment_Average reducer ...")
        self.acronym += "-z-" + str(self.zeta)

    def create(self, atomic_desc, atomic_numbers=[]):
        if self.element_wise:
            return self.acronym, Descriptor_By_Species(np.power(atomic_desc, self.zeta), atomic_numbers, self.species,
                                                       True)
        else:
            return self.acronym, np.mean(np.power(atomic_desc, self.zeta), axis=0)


class Atomic_2_Global_Moment_Sum(Atomic_2_Global_Base):
    """ 
    get the global descriptor from atomic ones 
    by averaging over the atomic descriptors of z th power 

    Parameters
    ----------
    zeta: take the zeta th power
    """

    def __init__(self, k_spec):

        super().__init__(k_spec)

        if "reducer_type" not in k_spec.keys() or k_spec["reducer_type"] != "moment_sum":
            raise ValueError("reducer type is not moment_sum or cannot find the type")

        try:
            self.zeta = k_spec['zeta']
        except:
            raise ValueError("cannot initialize the zeta list")

        print("Using Atomic_2_Global_Moment_Sum reducer ...")
        self.acronym += "-z-" + str(self.zeta) + "-sum"

    def create(self, atomic_desc, atomic_numbers=[]):
        if self.element_wise:
            return self.acronym, Descriptor_By_Species(np.power(atomic_desc, self.zeta), atomic_numbers, self.species,
                                                       False)
        else:
            return self.acronym, np.sum(np.power(atomic_desc, self.zeta), axis=0)


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
    for species in global_species:
        atomic_desc_by_species = [atomic_desc[i] for i, at in enumerate(atomic_numbers) if at == species]
        if average_over_natom and len(atomic_desc_by_species) > 0:
            # normalize by the number of atoms
            desc_by_species[species] = np.mean(atomic_desc_by_species, axis=0)
        elif len(atomic_desc_by_species) > 0:
            desc_by_species[species] = np.sum(atomic_desc_by_species, axis=0)
        else:
            desc_by_species[species] = 0
        # print(np.shape(atomic_desc),len(atomic_desc))

    desc_len = np.shape(atomic_desc)[1]
    desc = np.zeros(desc_len * len(global_species), dtype=float)
    for i, species in enumerate(global_species):
        desc[i * desc_len:(i + 1) * desc_len] = desc_by_species[species]
    return np.asarray(desc)
