import os
import numpy as np
from ase.io import read, write

class ASAPXYZ:
    def __init__(self, fxyz=None, stride=1):
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

        # store the xyz file
        self.frames = None
        self.nframes = 0
        self.natom_list = []
        self.total_natoms = 0

        if not os.path.isfile(self.fxyz):
            raise IOError('Cannot find the xyz file.')

        # try to read the xyz file
        try:
            self.frames = read(self.fxyz, slice(0,None,self.stride))
        except:
            raise ValueError('Exception occurred when loading the xyz file')


        self.nframes = len(self.frames)
        for frame in self.frames:
            # record the total number of atoms
            self.natom_list.append(len(frame.get_positions()))
        self.total_natoms = np.sum(self.natom_list)
        print('load xyz file: ', self.fxyz, 
              ', a total of ', str(self.nframes), 'frames',
              ', a total of ', str(self.total_natoms), 'atoms')

    def get_xyz(self):
        return self.frames

    def get_total_natoms(self):
        return self.total_natoms

    def get_natom_list(self):
        return self.natom_list

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
        ndesc = 0
        atom_index = 0
        [ found_desc, found_atomic_desc ] = [ False, False ]

        # load from xyz file
        if self.nframes > 1:
            for i, frame in enumerate(self.frames):
                if desc_name in frame.info:
                    try:
                        desc.append(frame.info[desc_name])
                        # make sure that each frame has a descriptor vector of the same length
                        if ndesc > 0 and len(frame.info[desc_name]) != ndesc:
                            raise ValueError('mismatch of number of descriptors between frames')
                        ndesc = len(frame.info[desc_name])
                        found_desc = True
                        #print("Found descriptor matrix from the xyz file")
                    except:
                        raise ValueError('The descriptor vectors of different frames are not of equal length')
                # for the atomic descriptors
                if use_atomic_desc:                    
                    try:
                        atomic_desc_now = frame.get_array(desc_name)
                        ndesc = np.shape(atomic_desc_now)[1]
                        if i == 0: 
                            atomic_desc = np.zeros((self.total_natoms, ndesc), dtype=float)
                            found_atomic_desc = True
                        if ndesc > 0 and np.shape(atomic_desc_now)[1] != ndesc:
                            raise ValueError('mismatch of number of descriptors between atoms')
                        natomnow = self.natom_list[i]
                        atomic_desc[atom_index:atom_index + natomnow, :] = atomic_desc_now
                        atom_index += natomnow                      
                        #print("Use atomic descriptor matrix with shape: ", np.shape(atomic_desc))
                    except:
                        raise IOError('Cannot find the atomic descriptors from the xyz file')
        else:
            # only one frame
            try:
                desc = self.frames[0].get_array(fmat)
            except:
                ValueError('Cannot read the descriptor matrix from single frame')

        if found_desc:
            #desc = np.asmatrix(desc)
            #desc.reshape((ndesc, self.nframes))
            print("Use descriptor matrix with shape: ", np.shape(desc))
        if found_atomic_desc:
            #atomic_desc = np.asmatrix(atomic_desc)
            #atomic_desc.reshape((ndesc, self.total_natoms))
            print("Use atomic descriptor matrix with shape: ", np.shape(atomic_desc))

        return desc, atomic_desc

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
        for i, frame in enumerate(self.frames):
            if desc_name in frame.info:
                frame.info[desc_name] = None

    def remove_atomic_descriptors(self, desc_name=None):
        """
        remove the desciptors
        """
        for i, frame in enumerate(self.frames):
            frame.set_array(desc_name, None)

    def write(self, filename):
        write(str(filename) + ".xyz", self.frames)

