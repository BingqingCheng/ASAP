"""
Methods and functions to compute global desciptors
"""
import numpy as np
import json
from ..io import NpEncoder, randomString
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
           So we can specify them as, e.g.
        {'global_desc2': 
        {"type": 'CM', "max_atoms" 30
        }}

        2. First compute an atomic descriptors (e.g. SOAP, ACSF,...) and convert to global ones
        e.g.
        {'global_desc2': {'atomic_descriptor': atomic_desc_dict, 'reducer_function': reducer_dict}}
        and
        atomic_desc_dict = {
        "firstsoap": 
        {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}
        }
        and
        reducer_dict = {'first_reducer': {'reducer_type': reducer_type,  
                          'zeta': zeta,
                          'species': species,
                          'element_wise': element_wise}}
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
        if "atomic_descriptor" in desc_spec.keys() and "reducer_function" in desc_spec.keys():
            return Global_Descriptor_from_Atomic(desc_spec)
        elif "type" not in desc_spec.keys():
            raise ValueError("Did not specify the type of the descriptor.")
        if desc_spec["type"] == "CM":
            return Global_Descriptor_CM(desc_spec)
        elif desc_spec["type"] == "MORGAN":
            return Global_Descriptor_Morgan(desc_spec)
        else:
            raise NotImplementedError

    def compute(self, frame):
        """
        compute the global descriptor vector and atomic descriptor matrix (if any) for a frame
        Parameters
        ----------
        frame: ASE atom object. Coordinates of a frame.

        Returns
        -------
        desc_dict: a dictionary. each entry contains the essential info of the descriptor, i.e. acronym 
                          and a np.array [N_desc]. Global descriptors for a frame.
                   e.g. {'d1':{ 'acronym': 'XXX', 'descriptors': `a np.array [N_desc]`}}
        atomic_desc_dict : a dictionary. each entry contains the essential info of the descriptor (acronym) 
                          and a np.array [N_desc*N_atoms]. Atomic descriptors for a frame.
                          e.g. {'ad1':{'acronym':'soap-1', 'atomic_descriptors': `a np.array [N_desc*N_atoms]`}}
        """
        global_desc_dict = {}
        atomic_desc_dict = {}
        for element in self.desc_spec_dict.keys():
            global_desc_dict[element], atomic_desc_dict[element] = self.engines[element].create(frame)
        return global_desc_dict, atomic_desc_dict


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
        # return the dictionaries for global descriptors and atomic descriptors (if any)
        return {'acronym': self.acronym, 'descriptors': []}, {}


class Global_Descriptor_from_Atomic(Global_Descriptor_Base):
    def __init__(self, desc_spec):
        """
        First compute an atomic descriptors (e.g. SOAP, ACSF,...) and convert to global ones

        Parameters
        ----------
        desc_spec: dictionaries that specify which global descriptor to use.

        e.g.
        {'global_desc2': {'atomic_descriptor': atomic_desc_dict, 'reducer_function': reducer_dict}}
        and
        atomic_desc_dict = {
        "firstsoap": 
        {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}
        }
        and
        reducer_dict = {'first_reducer': {'reducer_type': reducer_type,  
                          'zeta': zeta,
                          'species': species,
                          'element_wise': element_wise}}
        """

        self._is_atomic = True

        if "atomic_descriptor" not in desc_spec.keys() or "reducer_function" not in desc_spec.keys():
            raise ValueError("Need to specify both atomic descriptors and reducer functions to used")

        self.atomic_desc_spec = desc_spec['atomic_descriptor']
        self.reducer_spec = desc_spec['reducer_function']

        # pass down some key information
        if 'species' in desc_spec.keys():
            # add some system specific information to the list to descriptor specifications
            for element in self.atomic_desc_spec.keys():
                self.atomic_desc_spec[element]['species'] = desc_spec['species']
            for element in self.reducer_spec.keys():
                self.reducer_spec[element]['species'] = desc_spec['species']
        if 'periodic' in desc_spec.keys():
            for element in self.atomic_desc_spec.keys():
                self.atomic_desc_spec[element]['periodic'] = desc_spec['periodic']

        # initialize a Atomic_Descriptors object
        self.atomic_desc = Atomic_Descriptors(self.atomic_desc_spec)
        # initialize a Atomic_2_Global_Descriptors object
        self.atomic_2_global = Atomic_2_Global_Descriptors(self.reducer_spec)

        # print("Using Atomic_2_Global_Descriptors ...")

        self.acronym = "atomic-to-global-" + randomString(6)

    def pack(self):
        return {'atomic_descriptor': self.atomic_desc.pack(), 'reducer_function': atomic_2_global.pack()}

    def create(self, frame):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        frame: ASE atom object. Coordinates of a frame.

        Returns
        -------
        desc_dict: a dictionary. each entry contains the essential info of the descriptor, i.e. acronym 
                          and a np.array [N_desc]. Global descriptors for a frame.
                   e.g. {'d1':{ 'acronym': 'XXX', 'descriptors': `a np.array [N_desc]`}}
        atomic_desc_dict : a dictionary. each entry contains the essential info of the descriptor (acronym) 
                          and a np.array [N_desc*N_atoms]. Atomic descriptors for a frame.
                          e.g. {'ad1':{'acronym':'soap-1', 'atomic_descriptors': `a np.array [N_desc*N_atoms]`}}
        """
        # compute atomic descriptor
        atomic_desc_dict = self.atomic_desc.compute(frame)
        # compute global descriptor for the frame
        return self.atomic_2_global.compute(atomic_desc_dict, frame.get_atomic_numbers()), atomic_desc_dict


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
            raise ValueError("Coulomb Matrix cannot be used for periodic systems")

        self.cm = CoulombMatrix(self.max_atoms)
        print("Using CoulombMatrix ...")
        # make an acronym
        self.acronym = "CM" + "-" + str(self.max_atoms)

    def create(self, frame):
        """
        compute the CM descriptor vector for a frame
        Parameters
        ----------
        frame: ASE atom object. Coordinates of a frame.

        Returns
        -------
        desc_dict: a dictionary. each entry contains the essential info of the descriptor, i.e. acronym 
                          and a np.array [N_desc]. Global descriptors for a frame.
                   e.g. {'d1':{ 'acronym': 'CM-*', 'descriptors': `a np.array [N_desc]`}}
        atomic_desc_dict : {}
        """
        if len(frame.get_positions()) > self.max_atoms:
            raise ValueError('the size of the system is larger than the max_atoms of the CM descriptor')
        # notice that we return an empty dictionary for "atomic descriptors"
        return {'acronym': self.acronym, 'descriptors': self.cm.create(frame, n_jobs=1)}, {}


class Global_Descriptor_Morgan(Global_Descriptor_Base):
    def __init__(self, desc_spec):

        if "type" not in desc_spec.keys() or desc_spec["type"] != "MORGAN":
            raise ValueError("Type is not MORGAN or cannot find the type of the descriptor")

        # defaults
        if "length" in desc_spec.keys():
            self.length = desc_spec["length"]
        else:
            self.length = 1024

        if "radius" in desc_spec.keys():
            self.radius = desc_spec["radius"]
        else:
            self.radius = 3

        if 'periodic' in desc_spec.keys() and desc_spec['periodic'] == True:
            raise ValueError("Morgan Fingerprints cannot be used for periodic systems")

        print("Using Morgan Fingerprints ...")
        # make an acronym
        self.acronym = "MORGAN"

    def _get_smiles(self, frame):
        if "smiles" in frame.info:
            # print(frame.info['smiles'])
            return frame.info['smiles']
        elif "SMILES" in frame.info:
            return frame.info['SMILES']
        else:
            raise ValueError('Cannot parse the smile string from the frame.info.')

    def create(self, frame):
        """
        Returns
        -------
        """

        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

        smiles = self._get_smiles(frame)
        mol = MolFromSmiles(smiles)
        fps = GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.length)
        fps = np.array(fps, dtype='float64')

        return {'acronym': self.acronym, 'descriptors': fps}
