"""
Methods and functions to compute atomic desciptors
"""
import numpy as np
import json
from ..io import NpEncoder

class Atomic_Descriptors:
    def __init__(self, desc_spec_dict={}):
        """
        Object handing the specification and the computation of atomic descriptors
        Parameters
        ----------
        desc_spec_dict: dictionaries that specify which atomic descriptor to use 
        e.g.
        desc_spec_dict = {
        "firstsoap": 
        {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}
        }
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
                self.acronym += engine.get_acronym()
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
        if desc_spec["type"] == "SOAP":
            return Atomic_Descriptor_SOAP(desc_spec)
        else:
            raise NotImplementedError 

    def compute(self, frame):
        atomic_desc = self.engines[0].create(frame)
        for engine in self.engines[1:]:
            atomic_desc = np.append(atomic_desc, engine.create(frame), axis=1)
        return atomic_desc

class Atomic_Descriptor_Base:
    def __init__(self, desc_spec):
        self._is_atomic = True
        self.acronym = ""
        pass
    def is_atomic(self):
        return self._is_atomic
    def get_acronym(self):
        # we use an acronym for each descriptor, so it's easy to find it and refer to it
        return self.acronym
    def create(self, frame):
        pass

class Atomic_Descriptor_SOAP(Atomic_Descriptor_Base):
    def __init__(self, desc_spec):
        """
        make a DScribe SOAP object
        """

        from dscribe.descriptors import SOAP

        if "type" not in desc_spec.keys() or desc_spec["type"] != "SOAP":
            raise ValueError("Type is not SOAP or cannot find the type of the descriptor")

        # required
        try:
            self.species = desc_spec['species']
            self.cutoff = desc_spec['cutoff']
            self.g = desc_spec['atom_gaussian_width']
            self.n = desc_spec['n']
            self.l = desc_spec['l']
        except:
            raise ValueError("Not enough information to intialize the `Atomic_Descriptor_SOAP` object")

        # we have defaults here
        if 'rbf' in desc_spec.keys():
            self.rbf = desc_spec['rbf']
        else:
            self.rbf = 'gto'

        if 'crossover' in desc_spec.keys():
            self.crossover = bool(desc_spec['crossover'])
        else:
            self.crossover = False

        if 'periodic' in desc_spec.keys():
            self.periodic = bool(desc_spec['periodic'])
        else:
            self.periodic = True


        self.soap = SOAP(species=self.species, rcut=self.cutoff, nmax=self.n, lmax=self.l,
                                         sigma=self.g, rbf=self.rbf, crossover=self.crossover, average=False,
                                         periodic=self.periodic)
        # make an acronym
        self.acronym = "SOAP-n" + str(self.n) + "-l" + str(self.l) + "-c" + str(self.cutoff) + "-g" + str(self.g)

    def create(self, frame):
        return self.soap.create(frame, n_jobs=8)
