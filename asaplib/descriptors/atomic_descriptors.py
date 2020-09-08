"""
Methods and functions to compute atomic desciptors
"""
import json

import numpy as np

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
        if desc_spec["type"] == "ACSF":
            return Atomic_Descriptor_ACSF(desc_spec)
        if desc_spec["type"] == "LMBTR_K2":
            return Atomic_Descriptor_LMBTR_K2(desc_spec)
        if desc_spec["type"] == "LMBTR_K3":
            return Atomic_Descriptor_LMBTR_K3(desc_spec)
        if desc_spec["type"] == "FCHL19":
            return Atomic_Descriptor_FCHL19(desc_spec)
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
            atomic_desc_dict[element]['acronym'], atomic_desc_dict[element]['atomic_descriptors'] = self.engines[
                element].create(frame)
        return atomic_desc_dict


class Atomic_Descriptor_Base:
    def __init__(self, desc_spec):
        self._is_atomic = True
        self.acronym = ""
        pass

    def is_atomic(self):
        return self._is_atomic

    def get_acronym(self):
        """ 
        we use an acronym for each descriptor, so it's easy to find it and refer to it
        """
        return self.acronym

    def create(self, frame):
        """
        notice that we return the acronym here!!!
        """
        return self.acronym, []

    def _get_pbc(self, frame):
        return any(frame.get_pbc())

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

        print("Using SOAP Descriptors ...")

        # make an acronym
        self.acronym = "SOAP-n" + str(self.n) + "-l" + str(self.l) + "-c" + str(self.cutoff) + "-g" + str(self.g)

    def create(self, frame):
        from dscribe.descriptors import SOAP
        self.soap = SOAP(species=self.species, rcut=self.cutoff, nmax=self.n, lmax=self.l,
                         sigma=self.g, rbf=self.rbf, crossover=self.crossover, average='off',
                         periodic=self._get_pbc(frame))

        # notice that we return the acronym here!!!
        return self.acronym, self.soap.create(frame, n_jobs=1)


class Atomic_Descriptor_ACSF(Atomic_Descriptor_Base):
    def __init__(self, desc_spec):
        """
        make a DScribe ACSF object

        see: 
        https://singroup.github.io/dscribe/tutorials/acsf.html

        # template for an ACSF descriptor
        # currenly Dscribe only supports ASCF for finite system!
        """

        if "type" not in desc_spec.keys() or desc_spec["type"] != "ACSF":
            raise ValueError("Type is not ACSF or cannot find the type of the descriptor")

        if 'periodic' in desc_spec.keys():
            self.periodic = bool(desc_spec['periodic'])
        if self.periodic == True:
            raise ValueError("Warning: currently DScribe only supports ACSF for finite systems")

        from dscribe.descriptors import ACSF

        self.acsf_dict = {
            'g2_params': None,
            'g3_params': None,
            'g4_params': None,
            'g5_params': None}

        # required
        try:
            self.species = desc_spec['species']
            self.cutoff = desc_spec['cutoff']
        except:
            raise ValueError("Not enough information to intialize the `Atomic_Descriptor_ACF` object")

        # fill in the values
        for k, v in desc_spec.items():
            if k in self.acsf_dict.keys():
                if isinstance(v, list):
                    self.acsf_dict[k] = np.asarray(v)
                else:
                    self.acsf_dict[k] = v

        self.acsf = ACSF(species=self.species, rcut=self.cutoff, **self.acsf_dict, sparse=False)

        print("Using ACSF Descriptors ...")

        # make an acronym
        self.acronym = "ACSF-c" + str(self.cutoff)
        if self.acsf_dict['g2_params'] is not None: self.acronym += "-g2-" + str(len(self.acsf_dict['g2_params']))
        if self.acsf_dict['g3_params'] is not None: self.acronym += "-g3-" + str(len(self.acsf_dict['g3_params']))
        if self.acsf_dict['g4_params'] is not None: self.acronym += "-g4-" + str(len(self.acsf_dict['g4_params']))
        if self.acsf_dict['g5_params'] is not None: self.acronym += "-g5-" + str(len(self.acsf_dict['g5_params']))

    def create(self, frame):
        # notice that we return the acronym here!!!
        return self.acronym, self.acsf.create(frame, n_jobs=1)


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
            self.normalization = None  # or "l2_each"

        if 'normalize_gaussians' in desc_spec.keys():
            self.normalize_gaussians = desc_spec['normalize_gaussians']
        else:
            self.normalize_gaussians = "True"  # or False

        if 'periodic' in desc_spec.keys():
            self.periodic = bool(desc_spec['periodic'])
        else:
            self.periodic = True

    def create(self, frame):
        # notice that we return the acronym here!!!
        return self.acronym, self.lmbtr.create(frame, n_jobs=1)


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

        self.lmbtr = LMBTR(species=self.species, periodic=self.periodic, flatten=True,
                           normalize_gaussians=self.normalize_gaussians,
                           k2=self.k2)

        print("Using LMBTR-K2 Descriptors ...")

        # make an acronym
        self.acronym = "LMBTR-K2"  # perhaps add more info here


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

        self.lmbtr = LMBTR(species=self.species, periodic=self.periodic, flatten=True,
                           normalize_gaussians=self.normalize_gaussians,
                           k3=self.k3)

        print("Using LMBTR-K3 Descriptors ...")

        # make an acronym
        self.acronym = "LMBTR-K3"  # perhaps add more info here


class Atomic_Descriptor_FCHL19(Atomic_Descriptor_Base):
    """
    Generate the FCHL19 representation (https://doi.org/10.1063/1.5126701).
    Requires the developer version of the QML package, see
    https://www.qmlcode.org/installation.html for installation instructions.


    Parameters
    ----------
    :param nRs2: Number of gaussian basis functions in the two-body terms
    :type nRs2: integer
    :param nRs3: Number of gaussian basis functions in the three-body radial part
    :type nRs3: integer
    :param nFourier: Order of Fourier expansion
    :type nFourier: integer
    :param eta2: Precision in the gaussian basis functions in the two-body terms
    :type eta2: float
    :param eta3: Precision in the gaussian basis functions in the three-body radial part
    :type eta3: float
    :param zeta: Precision parameter of basis functions in the three-body angular part
    :type zeta: float
    :param two_body_decay: exponential decay for the two body function
    :type two_body_decay: float
    :param three_body_decay: exponential decay for the three body function
    :type three_body_decay: float
    :param three_body_weight: relative weight of the three body function
    :type three_body_weight: float
    :is_periodic: Boolean determining Whether the system is periodic.
    :type Boolean:
    """

    def __init__(self, desc_spec):

        if "type" not in desc_spec.keys() or desc_spec["type"] != "FCHL19":
            raise ValueError("Type is not FCHL19 or cannot find the type of the descriptor")

        if 'periodic' in desc_spec.keys():
            self.periodic = bool(desc_spec['periodic'])
        if self.periodic == True:
            raise ValueError("Warning: currently DScribe only supports FCHL19 for finite systems")

        print("Warning: This FCHL19 atomic descriptor is untested, because I (Bingqing) cannot install QML!")
        raise NotImplementedError

        self.fchl_acsf_dict = {'nRs2': None,
                               'nRs3': None,
                               'nFourier': None,
                               'eta2': None,
                               'eta3': None,
                               'zeta': None,
                               'rcut': None,
                               'acut': None,
                               'two_body_decay': None,
                               'three_body_decay': None,
                               'three_body_weight': None,
                               'pad': False, 'gradients': False}

        # required
        try:
            self.species = desc_spec['species']
        except:
            raise ValueError("Not enough information to intialize the `Atomic_Descriptor_ACF` object")

        # fill in the values
        for k, v in desc_spec.items():
            if k in self.fchl_acsf_dict.keys():
                self.fchl_acsf_dict[k] = v

        print("Using FCHL19 Descriptors ...")

        # make an acronym
        self.acronym = "FCHL19-c"  # add more stuff here

    def _repr_wrapper(frame, elements,
                      nRs2=24, nRs3=20,
                      nFourier=1, eta2=0.32, eta3=2.7,
                      zeta=np.pi, rcut=8.0, acut=8.0,
                      two_body_decay=1.8, three_body_decay=0.57,
                      three_body_weight=13.4, stride=1):

        nuclear_charges, coordinates = frame.get_atomic_numbers(), frame.get_positions()
        rep = generate_fchl_acsf(nuclear_charges, coordinates, elements,
                                 nRs2=nRs2, nRs3=nRs3, nFourier=nFourier,
                                 eta2=eta2, eta3=eta3, zeta=zeta,
                                 rcut=rcut, acut=acut,
                                 two_body_decay=two_body_decay, three_body_decay=three_body_decay,
                                 three_body_weight=three_body_weight,
                                 pad=False, gradients=False)
        rep_out = np.zeros((rep.shape[0], len(elements), rep.shape[1]))

        for i, z in enumerate(nuclear_charges):
            j = np.where(np.equal(z, elements))[0][0]
            rep_out[i, j] = rep[i]
        rep_out = rep_out.reshape(len(rep_out), -1)
        return rep_out

    def create(self, frame):
        # notice that we return the acronym here!!!
        return self.acronym, self_repr_wrapper(frame, self.species, **self.fchl_acsf_dict)
