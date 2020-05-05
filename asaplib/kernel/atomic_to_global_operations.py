"""
Methods and functions to obtain global descriptors from atomic descriptors
or the similarity/kernel between two structures using their atomic descriptors
"""

import numpy as np

def Avgerage_Descriptor(atomic_desc):
    """ get the global descriptor from atomic ones by a simple averaging

    Parameters
    ----------
    atomic_desc: np.matrix. [N_atoms, N_desc]. Atomic descriptors for a frame.

    Returns
    -------
    desc: np.matrix [N_desc]. Global descriptors for a frame.
    """
    return atomic_desc.mean(axis=0)

