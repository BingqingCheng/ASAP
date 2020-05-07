"""
Methods and functions to compute global desciptors
"""
import numpy as np
import json
from ..io import NpEncoder
from .atomic_to_global import Atomic_2_Global_Descriptors
from .atomic_descriptors import Atomic_Descriptors

class Global_Descriptors:
    def __init__(self, desc_spec_dict={}):
        """
        Object handing the specification and the computation of global descriptors
        global descriptors mean descriptors of a whole structure
        atomic descriptors mean descriptors of an atom centered environment inside a structure

        Parameters
        ----------
        desc_spec_dict: dictionaries that specify which global descriptor to use.
        We have two options here
        1. Some descriptors are already global in nature, e.g. the Coulomb Matrix, Morgan fingerprints, etc.
           So we can secify them as, e.g.
        {'global_desc2': 
        {"type": 'CM', "max_atoms" 30
        }}

        2. First compute an atomic descriptors (e.g. SOAP, ACSF,...) and convert to global ones
        e.g.
        {'global_desc2': {'atomic_descriptor': atomic_desc_dict, 'kernel_function': kernel_dict}}
        and
        atomic_desc_dict = {
        "firstsoap": 
        {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}
        }
        and
        kernel_dict = {'first_kernel': {'kernel_type': kernel_type,  
                          'zeta_list': zeta_list,
                          'species': species,
                          'element_wise': element_wise}}
        """

        self.desc_spec_dict = desc_spec_dict
        # list of Atomic_Descriptor objections
        self.engines = []
        self.acronym = ""

        self.bind()

    def add(self, desc_spec, tag):
        """
        adding the specifications of a new atomic descriptors
        Parameters
        ----------
        desc_spec: a dictionary that specify which atomic descriptor to use 
        """
        self.desc_spec_dict[tag] = desc_spec

    def pack(self):
        return json.dumps(self.desc_spec_dict, sort_keys=True, cls=NpEncoder)

    def get_acronym(self):
        if self.acronym == "":
            for engine in self.engines: 
                self.acronym.append(engine.get_acronym())
        return self.acronym

    def bind(self):
        """
        binds the objects that actually compute the descriptors
        these objects need to have .create(frame) method to compute the descriptors of frame (a xyz object)
        """
        # clear up the objects
        self.engines = []
        for element in self.desc_spec_dict.keys():
            self.engines.append(self._call(self.desc_spec_dict[element]))

    def _call(self, desc_spec):
        """
        call the specific descriptor objects
        """
        if "type" not in desc_spec.keys():
            raise ValueError("Did not specify the type of the descriptor.")
        if desc_spec["type"] == "CM":
            return Global_Descriptor_CM(desc_spec)
        else:
            raise NotImplementedError 

    def compute(self, frame):
        global_desc, atomic_desc = self.engines[0].create(frame)
        for engine in self.engines[1:]:
            global_desc_new, atomic_desc_new = self.engines[0].create(frame)
            global_desc = np.append(global_desc, global_desc_new, axis=0)
            atomic_desc = np.append(atomic_desc, atomic_desc_new, axis=0)
        del global_desc_new, atomic_desc_new
        return global_desc, atomic_desc

class Global_Descriptor_Base:
    def __init__(self, desc_spec):
        self._is_atomic = False
        self.acronym = ""
        pass
    def is_atomic(self):
        return self._is_atomic
    def get_acronym(self):
        # we use an acronym for each descriptor, so it's easy to find it and refer to it
        return self.acronym
    def create(self, frame):
        return [], []

class Global_Descriptor_from_Atomic(Global_Descriptor_Base):
    def __init__(self, desc_spec):
        """
        First compute an atomic descriptors (e.g. SOAP, ACSF,...) and convert to global ones

        Parameters
        ----------
        desc_spec: dictionaries that specify which global descriptor to use.

        e.g.
        {'global_desc2': {'atomic_descriptor': atomic_desc_dict, 'kernel_function': kernel_dict}}
        and
        atomic_desc_dict = {
        "firstsoap": 
        {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}
        }
        and
        kernel_dict = {'first_kernel': {'kernel_type': kernel_type,  
                          'zeta_list': zeta_list,
                          'species': species,
                          'element_wise': element_wise}}
        """

        self._is_atomic = True

        if "atomic_descriptor" not in desc_spec.keys() or "kernel_function" not in desc_spec.keys():
            raise ValueError("Need to specify both atomic descriptors and kernel functions to used")


        self.atomic_desc_spec = desc_spec['atomic_descriptor']

        self.kernel_spec = desc_spec['kernel_function']

        # pass down some key information
        if 'species' in desc_spec.keys():
            # add some system specific information to the list to descriptor specifications
            for element in self.atomic_desc_spec.keys():
                self.atomic_desc_spec[element]['species'] = self.global_species
                self.atomic_desc_spec[element]['periodic'] = self.periodic
            for element in self.kernel_spec.keys():
                self.kernel_spec[element]['species'] = self.global_species

        # initialize a Atomic_Descriptors object
        self.atomic_desc = Atomic_Descriptors(desc_spec_dict)
        # initialize a Atomic_2_Global_Descriptors object
        self.atomic_2_global = Atomic_2_Global_Descriptors(kernel_spec_dict)

        self.acronym = self.atomic_desc.get_acronym() + '-' + self.atomic_2_global.get_acronym()

    def pack(self):
        return {'atomic_descriptor': self.atomic_desc.pack(), 'kernel_function': atomic_2_global.pack())

    def create(self, frame):
        # compute atomic descriptor
        fnow = self.atomic_desc.compute(frame)
        # compute global descriptor for the frame
        return self.atomic_2_global.compute(fnow), fnow

class Global_Descriptor_CM(Global_Descriptor_Base):
    def __init__(self, desc_spec):
        """
        make a DScribe CM object
        """

        from dscribe.descriptors import CoulombMatrix

        if "type" not in desc_spec.keys() or desc_spec["type"] != "CM":
            raise ValueError("Type is not CM or cannot find the type of the descriptor")

        # required
        try:
            self.max_atoms = desc_spec['max_atoms']
        except:
            raise ValueError("Not enough information to intialize the `Atomic_Descriptor_CM` object")

        if 'periodic' in desc_spec.keys() and desc_spec['periodic'] == True:
            raise ValueError("CoulombMatrix cannot be used for periodic systems")


        self.cm = CoulombMatrix(max_atoms)
        # make an acronym
        self.acronym = "CM" + "-ma-" + str(max_atoms)

    def create(self, frame):
        # notice that we return an empty list for "atomic descriptors"
        return self.cm.create(frame, n_jobs=8), []
