import os
import json
import numpy as np
from ase.io import read, write
from ..io import randomString,  NpEncoder
from ..descriptors import Atomic_Descriptors, Atomic_2_Global_Descriptors

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
        self.computed_desc_dict = {}

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

    def save_descriptor_state(self, filename):
        with open(filename+'-descriptor-state.json', 'w') as jd:
            json.dump(self.computed_desc_dict, jd, sort_keys=True, cls=NpEncoder)

    def compute_atomic_descriptors(self, desc_spec_list={}, sbs=[], tag=None):
        """
        compute the atomic descriptors for selected frames
        Parameters
        ----------
        desc_spec: a list of dictionaries, contrains infos on the descriptors to use
        sbs: array, integer
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)
        if tag is None: tag = randomString(6)

        # add some system specific information to the list to descriptor specifications
        for element in desc_spec_dict.keys():
            desc_spec_dict[element]['species'] = self.global_species
            desc_spec_dict[element]['periodic'] = self.periodic

        # business!
        atomic_desc = Atomic_Descriptors(desc_spec_dict)
        desc_name_long = atomic_desc.pack()

        for i in sbs:
            frame = self.frames[sbs]
            frame.new_array(desc_name, atomic_desc.create(frame))

        # we mark down that this descriptor has been computed
        self.computed_desc_dict[tag] =  {'atomic_descriptor': desc_name_long, 'kernel_function': None}

    def compute_global_descriptors(self, desc_spec_dict={}, kernel_spec_dict={}, sbs=[], keep_atomic = False, tag=None, ktag=None):
        """
        compute the atomic descriptors for selected frames
        Parameters
        ----------
        desc_spec_dict: a list of dictionaries, contrains infos on the descriptors to use
        kernel_spec_dict: a dictionary contains specifications on how the global descriptors should be computed from the local ones
                     e.g. kernel_spec={'kernel1': {'kernel_type'='average','zeta_list'=[1,2,3],'element_wise'=True}}
        sbs: array, integer
        """

        if len(sbs) == 0:
            sbs = range(self.nframes)

        # add some system specific information to the list to descriptor specifications
        for element in desc_spec_dict.keys():
            desc_spec_dict[element]['species'] = self.global_species
            desc_spec_dict[element]['periodic'] = self.periodic
        for element in kernel_spec_dict.keys():
            kernel_spec_dict[element]['species'] = self.global_species

        # business!
        atomic_desc = Atomic_Descriptors(desc_spec_dict)
        desc_name_long = atomic_desc.pack()
        if tag is None: tag = atomic_desc.get_acronym() # '-'+randomString(6)

        atomic_2_global = Atomic_2_Global_Descriptors(kernel_spec_dict)
        kernel_name_long = atomic_2_global.pack()
        if ktag is None: ktag =atomic_2_global.get_acronym() # '-'+randomString(6)

        for i in sbs:
            frame = self.frames[i]
            # compute atomic descriptor
            fnow = atomic_desc.compute(frame)
            if keep_atomic:
                frame.new_array(tag, fnow)
            # compute global descriptor for the frame
            frame.info[tag+'-'+ktag] = atomic_2_global.compute(fnow)

        # we mark down that this descriptor has been computed
        self.computed_desc_dict[tag+'-'+ktag] = {'atomic_descriptor': desc_name_long, 'kernel_function': kernel_name_long}

    def get_descriptors(self, desc_name=None, use_atomic_desc=False):
        """ extract the descriptor array from each frame

        Parameters
        ----------
        fmat: string_like, the name of the descriptors in the extended xyz file
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
                desc = np.row_stack([a.info[desc_name] for a in self.frames])
                print("Use descriptor matrix with shape: ", np.shape(desc))
                # for the atomic descriptors
                if use_atomic_desc:
                    atomic_desc = np.concatenate([a.get_array(desc_name) for a in self.frames])
                    print("Use atomic descriptor matrix with shape: ", np.shape(atomic_desc))
            except:
                pass
        else:
            # only one frame
            try:
                desc = self.frames[0].get_array(desc_name)
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

    def write_descriptor_matrix(self, filename, desc_name, sbs=[], comment='#'):
        """
        write the selected descriptor matrix in a matrix format to file

        Parameters
        ----------
        filename: str
        desc_name: str. Name of the properties/descriptors to write
        sbs: array, integer
        comment: str
        """

        desc, _ = self.get_descriptors(desc_name, False, sbs)

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
