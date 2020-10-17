"""
ASAPXYZ class for handing atomic coordinate input and compute/output
"""
import os
import glob
import json
from yaml import dump as ydump
from yaml import Dumper
import numpy as np
from ase import Atoms
from ase.io import read, write
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import pandas as pd

from ..io import randomString, NpEncoder
from ..descriptors import Atomic_Descriptors, Global_Descriptors


class ASAPXYZ:
    """extended xyz class

    Parameters
    ----------
    fxyz: string_like
         the path to the extended xyz file
    fmat: string_like
           the name of the descriptors in the extended xyz file
    use_atomic_desc: bool
            return the descriptors for each atom, read from the xyz file
    stride: int
           the stride when reading the xyz file
    """

    def __init__(self, fxyz=None, stride=1, periodic=True, fileformat=None):
        # compile a list of matching xyz files
        # in fact they don't strictly need to be xyz format, anything that can be read by ASE is fine
        # a list of possible file formats: https://wiki.fysik.dtu.dk/ase/ase/io/io.html
        if '*' in fxyz:
            self.fxyz = glob.glob(fxyz)
            print("Find matching input files with coordinates: ", self.fxyz)
        else:
            self.fxyz = fxyz
        # essential
        self.stride = stride
        self.periodic = periodic
        if fileformat is not None:
            import ast
            self.fileformat = ast.literal_eval(fileformat)
        else:
            self.fileformat = {}

        # store the xyz file
        self.frames = None
        self.nframes = 0
        self.natom_list = []  # number of atoms for each frame
        self.total_natoms = 0  # total number of atoms for all frames
        self.global_species = []  # list of elements contains in all frames

        # record the state of the computation, e.g. which descriptors have been computed
        self.computed_desc_dict = {'data': {'fxyz': fxyz}}
        self.computed_desc_dict = {'descriptors': {}}
        # the conversion between tag of the descriptors and their acronyms
        self.tag_to_acronym = {'global': {}, 'atomic': {}}

        # we make a dictionary to store the computed descriptors
        self.global_desc = {}
        # this is for the atomic ones
        self.atomic_desc = {}

        # try to read the xyz file
        try:
            if isinstance(self.fxyz, (tuple, list)):
                self.frames = []
                for f in self.fxyz:
                    self.frames += read(f, slice(0, None, self.stride), **self.fileformat)
            else:
                self.frames = read(self.fxyz, slice(0, None, self.stride), **self.fileformat)
        except:
            raise ValueError('Exception occurred when loading the input file')

        self.nframes = len(self.frames)
        all_species = []
        for i, frame in enumerate(self.frames):
            # record the total number of atoms
            self.natom_list.append(len(frame.get_positions()))
            all_species.extend(frame.get_atomic_numbers())
            if not self.periodic or not np.sum(frame.get_cell()) > 0:
                frame.set_pbc([False, False, False])
            # we also initialize the descriptor dictionary for each frame
            self.global_desc[i] = {}
            self.atomic_desc[i] = {}

        self.total_natoms = np.sum(self.natom_list)
        self.max_atoms = max(self.natom_list)
        # Keep things in plain python for serialisation
        self.global_species = np.unique(all_species).tolist()
        print('load xyz file: ', self.fxyz,
              ', a total of ', str(self.nframes), 'frames',
              ', a total of ', str(self.total_natoms), 'atoms',
              ', with elements: ', self.global_species, '.')

    def get_xyz(self):
        return self.frames

    def get_num_frames(self):
        return self.nframes

    def get_total_natoms(self):
        return self.total_natoms

    def get_natom_list(self):
        return self.natom_list

    def get_global_species(self):
        return self.global_species

    def get_natom_list_by_species(self, species_name=None):
        if species_name is None:
            return self.natom_list
        elif species_name in self.global_species:
            return [a.get_atomic_numbers().tolist().count(species_name) for a in self.frames]
        else:
            raise ValueError("Cannot find the specified chemical species in the data set.")

    def save_state(self, filename, mode='yaml'):
        if mode == 'yaml':
            with open(filename + '-state.yaml', 'w') as yd:
                ydump(self.computed_desc_dict, yd, sort_keys=True, Dumper=Dumper)
        else:
            with open(filename + '-state.json', 'w') as jd:
                json.dump(self.computed_desc_dict, jd, sort_keys=True, cls=NpEncoder)

    def save_descriptor_acronym_state(self, filename, mode='yaml'):

        if mode == 'yaml':
            with open(filename + '-descriptor-acronyms.yaml', 'w') as yd:
                ydump(self.tag_to_acronym, yd, sort_keys=True)
        else:
            with open(filename + '-descriptor-acronyms.json', 'w') as jd:
                json.dump(self.tag_to_acronym, jd, sort_keys=True, cls=NpEncoder)

    def symmetrise(self, sbs=[], symprec=1e-2):
        import spglib
        if len(sbs) == 0:
            sbs = range(self.nframes)
        for i in sbs:
            frame = self.frames[i]
            frame.info['space_group'] = spglib.get_spacegroup(frame, symprec=symprec)

    def standardize(self, sbs=[], symprec=1e-2):
        """
        reduce to primitive cell
        """
        import spglib
        if len(sbs) == 0:
            sbs = range(self.nframes)
        for i in sbs:
            frame = self.frames[i]
            lattice, scaled_positions, numbers = spglib.standardize_cell(frame,
                                                                         to_primitive=1,
                                                                         no_idealize=1,
                                                                         symprec=symprec)
            self.frames[i] = Atoms(numbers=numbers, cell=lattice, scaled_positions=scaled_positions,
                                   pbc=frame.get_pbc())

    def _add_info_to_desc_spec(self, desc_spec_dict):
        """
        add some system specific information to the list to descriptor specifications
        Parameters
        ----------
        desc_spec_dict: dictionaries 
                        dict that specify which global descriptor to use.
        """
        for element in desc_spec_dict.keys():
            desc_spec_dict[element]['species'] = self.global_species
            desc_spec_dict[element]['periodic'] = self.periodic
            desc_spec_dict[element]['max_atoms'] = self.max_atoms

    def compute_atomic_descriptors(self, desc_spec_dict={}, sbs=[], tag=None, n_process=1):
        """
        compute the atomic descriptors for selected frames
        Parameters
        ----------
        desc_spec: a list of dictionaries
                 contrains infos on the descriptors to use
        e.g.
        .. code-block:: python

            atomic_desc_dict = {
            "firstsoap": 
            {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}}
        
        sbs: array, integer
             the index of the subset of structures to compute
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)
        if tag is None: tag = randomString(6)

        # add some system specific information to the list to descriptor specifications
        self._add_info_to_desc_spec(desc_spec_dict)

        # business!
        atomic_desc = Atomic_Descriptors(desc_spec_dict)

        # serial computation
        if n_process == 1:
            for i in tqdm(sbs):
                frame = self.frames[i]
                # compute atomic descriptor
                self.atomic_desc[i].update(atomic_desc.compute(frame))
        # parallel computation
        elif n_process >= 2:
            results = Parallel(n_jobs=n_process, verbose=1)(delayed(atomic_desc.compute)(self.frames[i]) for i in sbs)
            for i, atomic_desc_dict_now in enumerate(results):
                self.atomic_desc[i].update(atomic_desc_dict_now)
        else:
            raise ValueError("Please set the number of processes to be a positive integer.")

        # we mark down that this descriptor has been computed
        self.computed_desc_dict[tag] = atomic_desc.desc_spec_dict

    def compute_global_descriptors(self, desc_spec_dict={}, sbs=[], keep_atomic=False, tag=None, n_process=1):
        """
        compute the atomic descriptors for selected frames
        Parameters
        ----------
        desc_spec_dict: dictionaries that specify which global descriptor to use.

        e.g.
        .. code-block:: python

            {'global_desc1': 
                          {"type": 'CM'}}

            # or

            {'global_desc2': 
                          {'atomic_descriptor': 
                                       atomic_desc_dict, 
                           'reducer_function': 
                                       reducer_dict
                          }}
        
            atomic_desc_dict = {
                              "firstsoap": 
                                       {"type": 'SOAP',
                                       "species": [1, 6, 7, 8], 
                                       "cutoff": 2.0, 
                                       "atom_gaussian_width": 0.2, 
                                       "n": 4, 
                                       "l": 4}}
        
            reducer_dict = {'first_reducer': 
                                     {'reducer_type': reducer_type,  
                                     'zeta': zeta,
                                     'species': species,
                                     'element_wise': element_wise}}
        
        sbs: array, integer
             list of the indexes of the subset
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)
        if tag is None: tag = randomString(6)

        # add some system specific information to the list to descriptor specifications
        self._add_info_to_desc_spec(desc_spec_dict)

        # business! Intialize a Global_Descriptors object
        global_desc = Global_Descriptors(desc_spec_dict)

        # serial computation
        if n_process == 1:
            for i in tqdm(sbs):
                frame = self.frames[i]
                # compute atomic descriptor
                desc_dict_now, atomic_desc_dict_now = global_desc.compute(frame)
                self.global_desc[i].update(desc_dict_now)
                if keep_atomic:
                    for _, v in atomic_desc_dict_now.items():
                        self.atomic_desc[i].update(v)
        # parallel computation
        elif n_process >= 2:
            results = Parallel(n_jobs=n_process, verbose=1)(delayed(global_desc.compute)(self.frames[i]) for i in sbs)
            for i, (desc_dict_now, atomic_desc_dict_now) in enumerate(results):
                self.global_desc[i].update(desc_dict_now)
                if keep_atomic:
                    for _, v in atomic_desc_dict_now.items():
                        self.atomic_desc[i].update(v)
        else:
            raise ValueError("Please set the number of processes to be a positive integer.")

        # we mark down that this descriptor has been computed
        self.computed_desc_dict['descriptors'][tag] = global_desc.desc_spec_dict

    def fetch_computed_descriptors(self, desc_dict_keys=[], sbs=[]):
        """
        Fetch the computed descriptors for selected frames
        Parameters
        ----------
        desc_spec_keys: a list (str-like) of keys 
                    for which computed descriptors to fetch.
        sbs: array, integer

        Returns
        -------
        desc: np.matrix [n_frame, n_desc]
        """
        if len(sbs) == 0:
            sbs = list(range(self.nframes))
        return np.vstack([self._parse_computed_descriptors_singleframe(desc_dict_keys, i) for i in sbs])

    def fetch_computed_atomic_descriptors(self, desc_dict_keys=[], sbs=[]):
        """
        Fetch the computed atomic descriptors for selected frames
        Parameters
        ----------
        desc_spec_keys: a list (str-like) of keys
                    for which computed descriptors to fetch.
        sbs: array, integer

        Returns
        -------
        desc: np.matrix [n_atoms, n_desc]
        """
        if len(sbs) == 0:
            sbs = list(range(self.nframes))
        return np.vstack([self._parse_computed_atomic_descriptors_singleframe(desc_dict_keys, i) for i in sbs])

    def _parse_computed_descriptors_singleframe(self, desc_dict_keys=[], i=0):
        """return the global descriptor computed for frame i"""
        # TODO: use the nested dictionary search `extract_from_nested_dict` in ..io
        desc_array = np.array([])
        for e in desc_dict_keys:
            try:
                desc_array = np.append(desc_array, self.global_desc[i][e]['descriptors'])
            except:
                # if we use atomic to global descriptor, this is a nested dictionary
                for e2 in self.global_desc[i][e].keys():
                    for e3 in self.global_desc[i][e][e2].keys():
                        desc_array = np.append(desc_array, self.global_desc[i][e][e2][e3]['descriptors'])
        return desc_array

    def _parse_computed_atomic_descriptors_singleframe(self, desc_dict_keys=[], i=0):
        """return the atomic descriptor computed for frame i"""
        return np.hstack( self.atomic_desc[i][e]['atomic_descriptors'] for e in desc_dict_keys)

    def _write_computed_descriptors_to_xyz(self, desc_dict_now, frame):
        """  
        we recorded the computed descriptors to the xyz object
        we use acronym to record the entry in the extended xyz file, so it's much easier to ready by human
        """
        for e in desc_dict_now.keys():
            try:
                frame.info[desc_dict_now[e]['acronym']] = desc_dict_now[e]['descriptors']
                self.tag_to_acronym['global'][e] = desc_dict_now[e]['acronym']
            except:
                # if we use atomic to global descriptor, this is a nested dictionary
                self.tag_to_acronym['global'][e] = {}
                for e2 in desc_dict_now[e].keys():
                    self.tag_to_acronym['global'][e][e2] = {}
                    for e3 in desc_dict_now[e][e2].keys():
                        frame.info[desc_dict_now[e][e2][e3]['acronym']] = desc_dict_now[e][e2][e3]['descriptors']
                        self.tag_to_acronym['global'][e][e2][e3] = desc_dict_now[e][e2][e3]['acronym']

    def _write_computed_atomic_descriptors_to_xyz(self, atomic_desc_dict_now, frame):
        """  
        we recorded the computed descriptors to the xyz object
        we use acronym to record the entry in the extended xyz file, so it's much easier to ready by human
        """
        for e in atomic_desc_dict_now.keys():
            frame.new_array(atomic_desc_dict_now[e]['acronym'],
                            atomic_desc_dict_now[e]['atomic_descriptors'])
            self.tag_to_acronym['atomic'][e] = atomic_desc_dict_now[e]['acronym']

    def _desc_name_with_wild_card(self, desc_name_list, atomic_desc=False):
        """
        Use a wildcard when specifying the name of the descriptors
        """
        new_desc_name = []
        for desc_name in desc_name_list:
            # print("desc_name", desc_name)
            if desc_name == '*':
                import re
                possible_desc_prefix = ['SOAP', 'ACSF', 'LMBTR', 'FCHL19', 'CM', 'pca', 'skpca', 'umap', 'tsne']
                for pre in possible_desc_prefix:
                    if atomic_desc:
                        for key in self.frames[0].arrays.keys():
                            if re.search(pre + '.+', key):
                                new_desc_name.append(key)
                    else:
                        for key in self.frames[0].info.keys():
                            if re.search(pre + '.+', key):
                                new_desc_name.append(key)
            elif '*' in desc_name:
                import re
                if atomic_desc:
                    for key in self.frames[0].arrays.keys():
                        if re.search(desc_name.replace('*', '.+'), key):
                            new_desc_name.append(key)
                else:
                    for key in self.frames[0].info.keys():
                        if re.search(desc_name.replace('*', '.+'), key):
                            new_desc_name.append(key)
            else:
                new_desc_name.append(desc_name)
        return new_desc_name

    def get_descriptors(self, desc_name_list=[], use_atomic_desc=False, species_name=None):
        """ extract the descriptor array from each frame

        Parameters
        ----------
        desc_name_list: a list of strings
                        the name of the .info[] in the extended xyz file
        use_atomic_desc: bool
                         return the descriptors for each atom, read from the xyz file
        species_name: int
                      the atomic number of the species selected.
                      Only the desciptors of atoms of the specified specied will be returned.
                      species_name=None means all atoms are selected.
                      
        Returns
        -------
        desc: np.matrix
        atomic_desc: np.matrix
        """
        desc = []
        atomic_desc = []

        if isinstance(desc_name_list, str):
            desc_name_list = [desc_name_list]

        desc_name_list = self._desc_name_with_wild_card(desc_name_list)
        print("Find the following descriptor names that match the specifications: ", desc_name_list)

        # load from xyz file
        try:
            # retrieve the descriptor vectors --- both of these throw a ValueError if any are missing or are of wrong shape
            desc = np.hstack(
                np.vstack([a.info[desc_name] for a in self.frames]) for desc_name in desc_name_list)
            print("Use global descriptor matrix with shape: ", np.shape(desc))
            # get the atomic descriptors with the same name
            if use_atomic_desc:
                atomic_desc = self.get_atomic_descriptors(desc_name_list, species_name)
        except:
            print("Cannot find the specified descriptors from xyz")

        return desc, atomic_desc

    def get_atomic_descriptors(self, desc_name_list=[], species_name=None):
        """ extract the descriptor array from each frame

        Parameters
        ----------
        desc_name_list: a list of strings
                     the name of the .info[] in the extended xyz file
        species_name: int
                        the atomic number of the species selected.
                        Only the desciptors of atoms of the specified specied will be returned.
                    species_name=None means all atoms are selected.
                         
        Returns
        -------
        atomic_desc: np.matrix
        """

        atomic_desc = []

        if isinstance(desc_name_list, str):
            desc_name_list = [desc_name_list]

        desc_name_list = self._desc_name_with_wild_card(desc_name_list, True)
        print("Find the following atomic descriptor names that match the specifications: ", desc_name_list)

        # load from xyz file
        try:
            if species_name is None:
                atomic_desc = np.vstack(
                    np.concatenate([a.get_array(desc_name) for a in self.frames]) for desc_name in desc_name_list)
            elif species_name in self.global_species:
                atomic_desc = np.hstack(np.concatenate(
                    [self._get_atomic_descriptors_by_species(a, desc_name, species_name) for a in self.frames]) for
                                              desc_name in desc_name_list)
            else:
                raise ValueError("Cannot find the specified chemical species in the data set.")
            print("Use atomic descriptor matrix with shape: ", np.shape(atomic_desc))
        except:
            print("Cannot find the specified atomic descriptors from xyz")

        return atomic_desc

    def _get_atomic_descriptors_by_species(self, frame, desc_name, species_name=None):
        species_index = [i for i, s in enumerate(frame.get_atomic_numbers()) if s == species_name]
        return frame.get_array(desc_name)[species_index]

    def get_property(self, y_key=None, extensive=False, sbs=[]):
        """ extract specified property from selected frames

        Parameters
        ----------
        y_key: string_like
               the name of the property in the extended xyz file
        sbs: array, integer

        Returns
        -------
        y_all: array [N_samples]
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)

        y_all = []
        try:
            for i in sbs:
                frame = self.frames[i]
                if y_key == 'volume' or y_key == 'Volume':
                    y_all.append(frame.get_volume() / len(frame.get_positions()))
                elif y_key == 'size' or y_key == 'Size':
                    y_all.append(len(frame.get_positions()))
                elif y_key == 'index' or y_key == 'Index' or y_key == None:
                    y_all.append(i)
                elif isinstance(y_key, int) and int(y_key) in self.global_species:
                    # count the number of atoms of this specified chemical element
                    y_all.append(frame.get_atomic_numbers().tolist().count(int(y_key)))
                elif extensive:
                    y_all.append(frame.info[y_key] / len(frame.get_positions()))
                else:
                    y_all.append(frame.info[y_key])
        except:
            try:
                for i in sbs:
                    frame = self.frames[i]
                    if extensive:
                        # use the sum of atomic properties
                        y_all.append(np.sum(frame.get_array(y_key)))
                    else:
                        # use the average of atomic properties
                        y_all.append(np.mean(frame.get_array(y_key)))
            except:
                raise ValueError('Cannot load the property vector')
        if len(np.shape(y_all)) > 1:
            raise ValueError('The property from the xyz file has more than one column')
        return np.array(y_all)

    def get_atomic_property(self, y_key=None, extensive=False, sbs=[], species_name=None):
        """ extract the property array from each atom

        Parameters
        ----------
        y_key: string_like
             the name of the property in the extended xyz file
        sbs: array, integer
        specie: int
                 the atomic number of the species selected.
                      Only the properties of atoms of the specified specied will be returned.
                      species_name=None means all atoms are selected.
        Returns
        -------
        y_all: array [N_atoms]
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)

        y_all = []
        try:
            # y_all = np.concatenate([a.get_array(y_key) for a in self.frames[sbs]]) # this doesn't work ?!
            for i in sbs:
                frame = self.frames[i]
                if species_name is None:
                    y_all = np.append(y_all, frame.get_array(y_key))
                elif species_name in self.global_species:
                    y_all = np.append(y_all, self._get_atomic_descriptors_by_species(frame, y_key, species_name))
                else:
                    raise ValueError("Cannot find the specified chemical species in the data set.")
        except:
            try:
                for index, y in enumerate(self.get_property(y_key, extensive, sbs)):
                    y_all = np.append(y_all, y * np.ones(self.natom_list[index]))
                print("Cannot find the atomic properties, use the per-frame property instead")
            except:
                raise ValueError('Cannot load the property vector')
        if len(np.shape(y_all)) > 1:
            raise ValueError('The property from the xyz file has more than one column')
        return y_all

    def set_descriptors(self, desc=None, desc_name=None):
        """ write the descriptor array to the atom object

        Parameters
        ----------
        desc: np.matrix, shape=[n_descriptors, n_frames]

        Returns
        -------
        """
        # TODO: (maybe?) rename this into set_info. Need to change pca.py/kpca.py/... that uses this function
        # check if the length of the descriptor matrix is the same as the number of frames
        if len(desc) != self.nframes and self.nframes > 1:
            raise ValueError('The length of the descriptor matrix is not the same as the number of frames.')
        if len(desc) != self.total_natoms and self.nframes == 1:
            raise ValueError('The length of the descriptor matrix is not the same as the number of atoms.')

        if self.nframes > 1:
            for i, frame in enumerate(self.frames):
                frame.info[desc_name] = desc[i]
        else:
            self.frames[0].new_array(desc_name, desc)

    def set_atomic_descriptors(self, atomic_desc=None, atomic_desc_name=None, species_name=None):
        """ write the descriptor array to the atom object

        Parameters
        ----------
        desc: np.matrix, shape=[n_descriptors, n_atoms]

        Returns
        -------
        """
        # check if the length of the descriptor matrix is the same as the total number of atoms
        if species_name is None and len(atomic_desc) != self.total_natoms:
            raise ValueError('The length of the atomic descriptor matrix is not the same as the total number of atoms.')

        atom_index = 0

        if species_name is None:
            for i, frame in enumerate(self.frames):
                natomnow = self.natom_list[i]
                frame.new_array(atomic_desc_name, np.array(atomic_desc)[atom_index:atom_index + natomnow])
                atom_index += natomnow
        else:
            if atomic_desc.ndim == 1:
                n_desc = 1
            else:
                n_desc = np.shape(atomic_desc)[1]
            for i, frame in enumerate(self.frames):
                array_now = np.zeros((self.natom_list[i], n_desc), dtype=float)
                for j, s in enumerate(frame.get_atomic_numbers()):
                    if s == species_name:
                        array_now[j] = atomic_desc[atom_index]
                        atom_index += 1
                    else:
                        array_now[j] = np.nan
                frame.new_array(atomic_desc_name, np.array(array_now))

    def remove_descriptors(self, desc_name_list=[]):
        """
        remove the desciptors
        """
        if isinstance(desc_name_list, str):
            desc_name_list = [desc_name_list]

        desc_name_list = self._desc_name_with_wild_card(desc_name_list)
        print("removing the global descriptors from output xyz with the names: ", desc_name_list)

        for dn in desc_name_list:
            for frame in self.frames:
                if dn in frame.info:
                    del frame.info[dn]
                else:
                    pass
                    # print("Warning: Cannot parse desc_name "+str(dn)+" when remove_descriptors.")

    def remove_atomic_descriptors(self, desc_name_list=[]):
        """
        remove the desciptors
        """
        if isinstance(desc_name_list, str):
            desc_name_list = [desc_name_list]
        desc_name_list = self._desc_name_with_wild_card(desc_name_list, True)
        print("removing the atomic descriptors from output xyz with the names: ", desc_name_list)

        for dn in desc_name_list:
            for frame in self.frames:
                if dn in frame.arrays:
                    del frame.arrays[dn]
                else:
                    pass
                    # print("Warning: Cannot parse desc_name "+str(dn)+" when remove_descriptors.")

    def load_properties(self, filename, header='infer', prefix='X', **kwargs):
        """
        Load properties from a CSV file

        Read in the CSV file and save the columns to the `info` dictionary of the
        frames.

        Parameters
        ----------
        filename: str 
                Name of the CSV file.
        header: int
            Row number of the header. Defaults to use the first row unless explicit
          names for the columns are given
        """
        data = pd.read_csv(filename, header=header, prefix=prefix, **kwargs)

        for frame, (_, row) in zip(self.frames, data.iterrows()):
            frame.info.update(row.to_dict())

    def write(self, filename, sbs=[], save_acronym=False, wrap_output=True):
        """
        write the selected frames or all the frames to a xyz file

        Parameters
        ----------
        filename: str
        sbs: array, integer
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)

        # prepare for the output
        if os.path.isfile(str(filename) + ".xyz"):
            os.rename(str(filename) + ".xyz", "bck." + str(filename) + ".xyz")

        for i in sbs:
            if wrap_output: self.frames[i].wrap()
            self._write_computed_descriptors_to_xyz(self.global_desc[i], self.frames[i])
            self._write_computed_atomic_descriptors_to_xyz(self.atomic_desc[i], self.frames[i])
            write(str(filename) + ".xyz", self.frames[i], append=True)

        # this acronym state file lets us know how the descriptors correspond to the outputs in the xyz file
        if save_acronym:
            self.save_descriptor_acronym_state(filename)

    def write_chemiscope(self, filename, sbs=None, save_acronym=False, cutoff=None, wrap_output=True):
        """
        write the selected frames or all the frames to ChemiScope JSON

        Parameters
        ----------
        filename: str
        sbs: array, integer
        cutoff: float
                generate cutoff for atomic environments, set to None to disable atomic environments
        """

        from asaplib.io.cscope import write_chemiscope_input

        if sbs is None:
            sbs = range(self.nframes)

        # prepare for the output
        if os.path.isfile(str(filename) + ".xyz"):
            os.rename(str(filename) + ".xyz", "bck." + str(filename) + ".xyz")

        for i in sbs:
            if wrap_output: self.frames[i].wrap()
            self._write_computed_descriptors_to_xyz(self.global_desc[i], self.frames[i])
            self._write_computed_atomic_descriptors_to_xyz(self.atomic_desc[i], self.frames[i])

        # this acronym state file lets us know how the descriptors correspond to the outputs in the xyz file
        if save_acronym:
            self.save_descriptor_acronym_state(filename)

        # Disable atomic environments if there isn't any data
        write_chemiscope_input(filename + '.json.gz', self.frames, cutoff=cutoff)

    def write_descriptor_matrix(self, filename, desc_name_list, sbs=[], comment=''):
        """
        write the selected descriptor matrix in a matrix format to file

        Parameters
        ----------
        filename: str
        desc_name_list: a list of str. 
                Name of the properties/descriptors to write
        sbs: array, integer
        comment: str
        """
        if len(sbs) == 0:
            sbs = range(self.nframes)

        desc, _ = self.get_descriptors(desc_name_list, False, sbs)

        if os.path.isfile(str(filename) + ".desc"):
            os.rename(str(filename) + ".desc", "bck." + str(filename) + ".desc")
        np.savetxt(str(filename) + ".desc", desc, fmt='%4.8f', header=comment)

    def write_atomic_descriptor_matrix(self, filename, desc_name, sbs=[], comment=''):
        """
        write the selected descriptor matrix in a matrix format to file

        Parameters
        ----------
        filename: str
        desc_name: str
                Name of the properties/descriptors to write
        sbs: array, integer
        comment: str
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)

        _, atomic_desc = self.get_descriptors(desc_name, True, sbs)

        if os.path.isfile(str(filename) + ".atomic-desc"):
            os.rename(str(filename) + ".atomic-desc", "bck." + str(filename) + "atomic-desc")
        np.savetxt(str(filename) + ".atomic-desc", desc, fmt='%4.8f', header=comment)

    def write_computed_descriptors(self, filename, desc_dict_keys=[], sbs=[], comment=''):
        """
        write the computed descriptors for selected frames
        Parameters
        ----------
        desc_spec_keys: list
              a list (str-like) of keys for which computed descriptors to fetch.
        sbs: array, integer

        Returns
        -------
        desc: np.matrix [n_frame, n_desc]
        """
        if len(sbs) == 0:
            sbs = range(self.nframes)
        np.savetxt(str(filename) + ".desc", self.fetch_computed_descriptors(desc_dict_keys, sbs), fmt='%4.8f',
                   header=comment)
