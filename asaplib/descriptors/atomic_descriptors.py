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
        self.engines = {}
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
            for element in self.desc_spec_dict.keys():
                self.acronym += self.engines[element].get_acronym()
        return self.acronym

    def bind(self):
        """
        binds the objects that actually compute the descriptors
        these objects need to have .create(frame) method to compute the descriptors of frame (a xyz object)
        """
        # clear up the objects
        self.engines = {}
        for element in self.desc_spec_dict.keys():
            self.engines[element] = self._call(self.desc_spec_dict[element])
            self.desc_spec_dict[element]['acronym'] = self.engines[element].get_acronym()

    def _call(self, desc_spec):
        """
        call the specific descriptor objects
        """
        if "type" not in desc_spec.keys():
            raise ValueError("Did not specify the type of the descriptor.")
        if desc_spec["type"] == "SOAP":
            return Atomic_Descriptor_SOAP(desc_spec)
        if desc_spec["type"] == "LMBTR_K2":
            return Atomic_Descriptor_LMBTR_K2(desc_spec)
        if desc_spec["type"] == "LMBTR_K3":
            return Atomic_Descriptor_LMBTR_K3(desc_spec)
        else:
            raise NotImplementedError 

    def compute(self, frame):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        frame: ASE atom object. Coordinates of a frame.

        Returns
        -------
        atomic_desc_dict : a dictionary. each entry contains the essential info of the descriptor (acronym) 
                          and a np.array [N_desc*N_atoms]. Atomic descriptors for a frame.
                          e.g. {'ad1':{'acronym':'soap-1', 'atomic_descriptors': `a np.array [N_desc*N_atoms]`}}
        """
        atomic_desc_dict = {}
        for element in self.desc_spec_dict.keys():
            atomic_desc_dict[element] = {}
            atomic_desc_dict[element]['acronym'], atomic_desc_dict[element]['atomic_descriptors'] = self.engines[element].create(frame)
        return atomic_desc_dict

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
        # notice that we return the acronym here!!!
        return self.acronym, []

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

        print("Using SOAP Descriptors ...")

        # make an acronym
        self.acronym = "SOAP-n" + str(self.n) + "-l" + str(self.l) + "-c" + str(self.cutoff) + "-g" + str(self.g)

    def create(self, frame):
        # notice that we return the acronym here!!!
        return self.acronym, self.soap.create(frame, n_jobs=8)


class Atomic_Descriptor_LMBTR(Atomic_Descriptor_Base):
    def __init__(self, desc_spec):
        """
        make a DScribe LMBTR object
        (see https://singroup.github.io/dscribe/tutorials/lmbtr.html)')
        Args:
            species:
            periodic (bool): Determines whether the system is considered to be
                periodic.
            k2 (dict): Dictionary containing the setup for the k=2 term.
                Contains setup for the used geometry function, discretization and
                weighting function. For example::

                    k2 = {
                        "geometry": {"function": "inverse_distance"},
                        "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 50},
                        "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                    }

            k3 (dict): Dictionary containing the setup for the k=3 term.
                Contains setup for the used geometry function, discretization and
                weighting function. For example::

                    k3 = {
                        "geometry": {"function": "angle"},
                        "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
                        "weighting" = {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                    }
            normalize_gaussians (bool): Determines whether the gaussians are
                normalized to an area of 1. Defaults to True. If False, the
                normalization factor is dropped and the gaussians have the form.
                :math:`e^{-(x-\mu)^2/2\sigma^2}`
            normalization (str): Determines the method for normalizing the
                output. The available options are:

                * "none": No normalization.
                * "l2_each": Normalize the Euclidean length of each k-term
                  individually to unity.

            flatten (bool): Whether the output should be flattened to a 1D
                array. If False, a dictionary of the different tensors is
                provided, containing the values under keys: "k1", "k2", and
                "k3":
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """

        # required
        try:
            self.species = desc_spec['species']
        except:
            raise ValueError("Not enough information to intialize the `Atomic_Descriptor_LMBTR` object")

        # we have defaults here
        if 'normalization' in desc_spec.keys():
            self.normalization = desc_spec['normalization']
        else:
            self.normalization =  None # or "l2_each"

        if 'normalize_gaussians' in desc_spec.keys():
            self.normalize_gaussians = desc_spec['normalize_gaussians']
        else:
            self.normalize_gaussians = "True" # or False

        if 'periodic' in desc_spec.keys():
            self.periodic = bool(desc_spec['periodic'])
        else:
            self.periodic = True

    def create(self, frame):
        # notice that we return the acronym here!!!
        return self.acronym, self.lmbtr.create(frame, n_jobs=8)

class Atomic_Descriptor_LMBTR_K2(Atomic_Descriptor_LMBTR):
    def __init__(self, desc_spec):

        super().__init__(desc_spec)

        from dscribe.descriptors import LMBTR

        if "type" not in desc_spec.keys() or desc_spec["type"] != "LMBTR_K2":
            raise ValueError("Type is not LMBTR_K2 or cannot find the type of the descriptor")

        # required
        try:
            self.k2 = desc_spec['k2']
        except:
            raise ValueError("Not enough information to intialize the `Atomic_Descriptor_LMBTR` object")

        self.lmbtr = LMBTR(species=self.species, periodic=self.periodic, flatten=True, normalize_gaussians=self.normalize_gaussians,
                            k2=self.k2)

        print("Using LMBTR-K2 Descriptors ...")

        # make an acronym
        self.acronym = "LMBTR-K2" # perhaps add more info here

class Atomic_Descriptor_LMBTR_K3(Atomic_Descriptor_LMBTR):
    def __init__(self, desc_spec):

        super().__init__(desc_spec)

        from dscribe.descriptors import LMBTR

        if "type" not in desc_spec.keys() or desc_spec["type"] != "LMBTR_K3":
            raise ValueError("Type is not LMBTR_K3 or cannot find the type of the descriptor")

        # required
        try:
            self.k2 = desc_spec['k3']
        except:
            raise ValueError("Not enough information to intialize the `Atomic_Descriptor_LMBTR` object")

        self.lmbtr = LMBTR(species=self.species, periodic=self.periodic, flatten=True, normalize_gaussians=self.normalize_gaussians,
                            k3=self.k3)

        print("Using LMBTR-K3 Descriptors ...")

        # make an acronym
        self.acronym = "LMBTR-K3" # perhaps add more info here
