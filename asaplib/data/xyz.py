import os
import json
import numpy as np
from ase.io import read, write
from ..io import randomString,  NpEncoder
from ..descriptors import Atomic_Descriptors, Global_Descriptors

class ASAPXYZ:
    def __init__(self, fxyz=None, stride=1, periodic=True):
        """extended xyz class

        Parameters
        ----------
        fxyz: string_like, the path to the extended xyz file
        fmat: string_like, the name of the descriptors in the extended xyz file
        use_atomic_desc: bool, return the descriptors for each atom, read from the xyz file
        stride: int, the stride when reading the xyz file
        """

        # essential
        self.fxyz = fxyz
        self.stride = stride
        self.periodic = periodic

        # store the xyz file
        self.frames = None
        self.nframes = 0
        self.natom_list = []
        self.total_natoms = 0
        self.global_species = []
        self.computed_desc_dict = {'data' : {'fxyz': fxyz} }
        self.tag_to_acronym = {'global':{}, 'atomic':{}}

        if not os.path.isfile(self.fxyz):
            raise IOError('Cannot find the xyz file.')

        # try to read the xyz file
        try:
            self.frames = read(self.fxyz, slice(0, None, self.stride))
        except:
            raise ValueError('Exception occurred when loading the xyz file')

        self.nframes = len(self.frames)
        all_species = []
        for frame in self.frames:
            # record the total number of atoms
            self.natom_list.append(len(frame.get_positions()))
            all_species.extend(frame.get_atomic_numbers())
            if not self.periodic or not np.sum(frame.get_cell()) > 0:
                frame.set_pbc([False, False, False])

        self.total_natoms = np.sum(self.natom_list)
        self.max_atoms = max(self.natom_list)
        self.global_species = np.unique(all_species)
        print('load xyz file: ', self.fxyz,
              ', a total of ', str(self.nframes), 'frames',
              ', a total of ', str(self.total_natoms), 'atoms',
              ', with elements: ', self.global_species,'.')

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

    def save_state(self, filename):
        with open(filename+'-state.json', 'w') as jd:
            json.dump(self.computed_desc_dict, jd, sort_keys=True, cls=NpEncoder)

    def save_descriptor_state(self, filename):
        with open(filename+'-descriptor-acronyms.json', 'w') as jd:
            json.dump(self.tag_to_acronym, jd, sort_keys=True, cls=NpEncoder)

    def compute_atomic_descriptors(self, desc_spec_list={}, sbs=[], tag=None):
        """
        compute the atomic descriptors for selected frames
        Parameters
        ----------
        desc_spec: a list of dictionaries, contrains infos on the descriptors to use
        e.g.
        atomic_desc_dict = {
        "firstsoap": 
        {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}
        }

        sbs: array, integer
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)
        if tag is None: tag = randomString(6)

        # add some system specific information to the list to descriptor specifications
        for element in desc_spec_dict.keys():
            desc_spec_dict[element]['species'] = self.global_species
            desc_spec_dict[element]['periodic'] = self.periodic
            desc_spec_dict[element]['max_atoms'] = self.max_atoms

        # business!
        atomic_desc = Atomic_Descriptors(desc_spec_dict)

        for i in sbs:
            frame = self.frames[sbs]
            atomic_desc_dict_now = atomic_desc.create(frame)
            self._parse_computed_atomic_descriptors(atomic_desc_dict_now, frame)

        # we mark down that this descriptor has been computed
        self.computed_desc_dict[tag] =  atomic_desc.pack()

    def compute_global_descriptors(self, desc_spec_dict={}, sbs=[], keep_atomic = False, tag = None):
        """
        compute the atomic descriptors for selected frames
        Parameters
        ----------
        desc_spec_dict: dictionaries that specify which global descriptor to use.

        e.g.
        {'global_desc1': 
        {"type": 'CM'}}

        e.g.
        {'global_desc2': {'atomic_descriptor': atomic_desc_dict, 'kernel_function': kernel_dict}}
        and
        atomic_desc_dict = {
        "firstsoap": 
        {"type": 'SOAP',"species": [1, 6, 7, 8], "cutoff": 2.0, "atom_gaussian_width": 0.2, "n": 4, "l": 4}
        }
        and
        kernel_dict = {'first_kernel': {'kernel_type': kernel_type,  
                          'zeta': zeta,
                          'species': species,
                          'element_wise': element_wise}}

        sbs: array, integer
        """



        if len(sbs) == 0:
            sbs = range(self.nframes)
        if tag is None: tag = randomString(6)

        # add some system specific information to the list to descriptor specifications
        for element in desc_spec_dict.keys():
            desc_spec_dict[element]['species'] = self.global_species
            desc_spec_dict[element]['periodic'] = self.periodic
            desc_spec_dict[element]['max_atoms'] = self.max_atoms

        # business!
        global_desc = Global_Descriptors(desc_spec_dict)

        for i in sbs:
            frame = self.frames[i]
            # compute atomic descriptor
            desc_dict_now, atomic_desc_dict_now = global_desc.compute(frame)
            self._parse_computed_descriptors(desc_dict_now, frame)
            if keep_atomic:
                self._parse_computed_atomic_descriptors(atomic_desc_dict_now, frame)
        # we mark down that this descriptor has been computed
        self.computed_desc_dict[tag] = global_desc.pack()

    def _parse_computed_descriptors(self, desc_dict_now, frame):
        """parse the information computed"""
        # now we recorded the computed descriptors to the xyz object
        for e in desc_dict_now.keys():
            # we use acronym to record the entry in the extended xyz file, so it's much easier to ready by human
            try:
                frame.info[desc_dict_now[e]['acronym']] = desc_dict_now[e]['descriptors']
                self.tag_to_acronym['global'][e] = desc_dict_now[e]['acronym']
            except:
                # if we use atomic to global descriptor, this is a nested dictionary
                self.tag_to_acronym['global'][e] = {}
                for e2 in desc_dict_now[e].keys():
                    self.tag_to_acronym['global'][e][e2] = {}
                    #print(e2), print(desc_dict_now[e][e2])
                    for e3 in desc_dict_now[e][e2].keys():
                        #print(e3); print(desc_dict_now[e][e2][e3]); print(desc_dict_now[e][e2][e3].keys())
                        frame.info[desc_dict_now[e][e2][e3]['acronym']] = desc_dict_now[e][e2][e3]['descriptors']
                        self.tag_to_acronym['global'][e][e2][e3] = desc_dict_now[e][e2][e3]['acronym']

    def _parse_computed_atomic_descriptors(self, atomic_desc_dict_now, frame):
        """parse the information computed"""
        #print(atomic_desc_dict_now)
        for e in atomic_desc_dict_now.keys():
            self.tag_to_acronym['atomic'][e] = {}
            for e2 in atomic_desc_dict_now[e].keys():
                frame.new_array(atomic_desc_dict_now[e][e2]['acronym'], atomic_desc_dict_now[e][e2]['atomic_descriptors'])
                self.tag_to_acronym['atomic'][e][e2] = atomic_desc_dict_now[e][e2]['acronym']

    def get_descriptors(self, desc_name_list=[], use_atomic_desc=False):
        """ extract the descriptor array from each frame

        Parameters
        ----------
        desc_name_list: a list of strings, the name of the descriptors in the extended xyz file
        use_atomic_desc: bool, return the descriptors for each atom, read from the xyz file

        Returns
        -------
        desc: np.matrix
        atomic_desc: np.matrix
        """
        desc = []
        atomic_desc = []

        # load from xyz file
        if self.nframes > 1:
            try:
                # retrieve the descriptor vectors --- both of these throw a ValueError if any are missing or are of wrong shape
                desc = np.column_stack(np.row_stack([a.info[desc_name] for a in self.frames]) for desc_name in desc_name_list)
                print("Use descriptor matrix with shape: ", np.shape(desc))
                # for the atomic descriptors
                if use_atomic_desc:
                    atomic_desc = np.column_stack(np.concatenate([a.get_array(desc_name) for a in self.frames]) for desc_name in desc_name_list)
                    print("Use atomic descriptor matrix with shape: ", np.shape(atomic_desc))
            except:
                pass
        else:
            # only one frame
            try:
                desc = np.column_stack(self.frames[0].get_array(desc_name) for desc_name in desc_name_list)
            except:
                ValueError('Cannot read the descriptor matrix from single frame')

        return desc, atomic_desc

    def get_property(self, y_key=None, extensive=False, sbs=[]):
        """ extract the descriptor array from each frame

        Parameters
        ----------
        y_name: string_like, the name of the property in the extended xyz file
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

    def get_atomic_property(self, y_key=None, extensive=False, sbs=[]):
        """ extract the property array from each atom

        Parameters
        ----------
        y_name: string_like, the name of the property in the extended xyz file
        sbs: array, integer

        Returns
        -------
        y_all: array [N_atoms]
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)

        y_all = []
        try:
            y_all = np.concatenate([a.get_array(y_key) for a in self.frames[sbs]])
        except:
            try:
                for index, y in enumerate(self.get_property(y_key, extensive, sbs)):
                    y_all = np.append(y_all, y * np.ones(self.natom_list[index]))
                print("Cannot find the atomic properties, use the per-frame property instead")
            except: raise ValueError('Cannot load the property vector')
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

    def set_atomic_descriptors(self, atomic_desc=None, atomic_desc_name=None):
        """ write the descriptor array to the atom object

        Parameters
        ----------
        desc: np.matrix, shape=[n_descriptors, n_atoms]

        Returns
        -------
        """
        # check if the length of the descriptor matrix is the same as the total number of atoms
        if len(atomic_desc) != self.total_natoms:
            raise ValueError('The length of the atomic descriptor matrix is not the same as the total number of atoms.')

        atom_index = 0
        for i, frame in enumerate(self.frames):
            natomnow = self.natom_list[i]
            frame.new_array(atomic_desc_name, atomic_desc[atom_index:atom_index + natomnow, :])
            atom_index += natomnow

    def remove_descriptors(self, desc_name=None):
        """
        remove the desciptors
        """
        for frame in self.frames:
            if desc_name in frame.info:
                del frame.info[desc_name]

    def remove_atomic_descriptors(self, desc_name=None):
        """
        remove the desciptors
        """
        for frame in self.frames:
            del frame.arrays[desc_name]

    def write(self, filename, sbs=[]):
        """
        write the selected frames or all the frames to a xyz file

        Parameters
        ----------
        filename: str
        sbs: array, integer
        """
        # prepare for the output
        if os.path.isfile(str(filename) + ".xyz"): 
            os.rename(str(filename) + ".xyz", "bck." + str(filename) + ".xyz")

        if len(sbs) > 0:
            for i in sbs:
                write(str(filename) + ".xyz", self.frames[i], append=True)
        else:
            write(str(filename) + ".xyz", self.frames)

    def write_descriptor_matrix(self, filename, desc_name_list, sbs=[], comment='#'):
        """
        write the selected descriptor matrix in a matrix format to file

        Parameters
        ----------
        filename: str
        desc_name_list: a list of str. Name of the properties/descriptors to write
        sbs: array, integer
        comment: str
        """

        desc, _ = self.get_descriptors(desc_name_list, False, sbs)

        if os.path.isfile(str(filename) + ".desc"): 
            os.rename(str(filename) + ".desc", "bck." + str(filename) + ".desc")
        np.savetxt(str(filename) + ".desc", desc, fmt='%4.8f', header=comment)

    def write_atomic_descriptor_matrix(self, filename, desc_name, sbs=[], comment='#'):
        """
        write the selected descriptor matrix in a matrix format to file

        Parameters
        ----------
        filename: str
        desc_name: str. Name of the properties/descriptors to write
        sbs: array, integer
        comment: str
        """

        _, atomic_desc = self.get_descriptors(desc_name, True, sbs)

        if os.path.isfile(str(filename) + "atomic-desc"): 
            os.rename(str(filename) + "atomic-desc", "bck." + str(filename) + "atomic-desc")
        np.savetxt(str(filename) + "atomic-desc", desc, fmt='%4.8f', header=comment)
