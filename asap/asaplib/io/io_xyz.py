"""
TODO: Module-level description
"""

import numpy as np
import math

zmap = {"H": 1,"He": 2,"Li": 3,"Be": 4,"B": 5,"C": 6,"N": 7,"O": 8,"F": 9,"Ne": 10,"Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15,"S": 16,"Cl": 17,"Ar": 18,"K": 19,"Ca": 20,"Sc": 21,"Ti": 22,"V": 23,"Cr": 24,"Mn": 25,"Fe": 26, "Ni": 28,"Ga": 31, "As":33}

"""
e.g.
   while True:
     try:
       [ na, cell, names, pos] = read_frame(ixyz)
     except: sys.exit(0)
"""


def read_xyz(filedesc):
  natoms = int(filedesc.readline())
  comment = filedesc.readline()

  cell = np.zeros(3, float)
  names = np.zeros(natoms, np.dtype('|S6'))
  q = np.zeros((natoms, 3), float)
  cell[:] = comment.split()[0:3]
  print(cell)
  for i in range(natoms):
    line = filedesc.readline().split()
    names[i] = line[0]
    q[i] = line[1:4]
    #print q[i]
  return [natoms, cell, names, q]


def write_ipixyz(outfile, cell, names, q):
    # output
    # compute the angles
    # mode is 'abcABC', then 'cell' takes an array of 6 floats
    # the first three being the length of the sides of the system parallelopiped, and the last three being the angles (in degrees) between those sides.
    # Angle A corresponds to the angle between sides b and c, and so on for B and C.
    supercell = np.zeros(3, float)
    angles = np.zeros(3, float)
    natom = len(q)

    for i in range(3):
        supercell[i] = np.linalg.norm(cell[i, :])

    angles[0] = np.arccos(np.dot(cell[1],cell[2])/supercell[1]/supercell[2])/math.pi*180.
    angles[1] = np.arccos(np.dot(cell[0],cell[2])/supercell[0]/supercell[2])/math.pi*180.
    angles[2] = np.arccos(np.dot(cell[0],cell[1])/supercell[0]/supercell[1])/math.pi*180.

    # write
    outfile.write("%d\n# CELL(abcABC):     %4.8f     %4.8f     %4.8f     %4.5f     %4.5f     %4.5f   cell{angstrom}  Traj: positions{angstrom}\n" % (natom,supercell[0],supercell[1],supercell[2],angles[0],angles[1],angles[2]))
    for i, qi in enumerate(q):
        #print (names[i],q[i*3],q[i*3+1],q[i*3+2])
        outfile.write("%s     %4.8f     %4.8f     %4.8f\n" % (names[i], qi[0], qi[1], qi[2]))
    return 0


def pbcdist(q1, q2, h, ih): 
      s = np.dot(ih, q1-q2)
      for i in range(3):
          s[i] -= round(s[i])
      return np.dot(h, s)


def h2abc(h):
    """Returns a description of the cell in terms of the length of the
       lattice vectors and the angles between them in radians.

    Takes the representation of the system box in terms of an upper triangular
    matrix of column vectors, and returns the representation in terms of the
    lattice vector lengths and the angles between them in radians.

    Args:
       h: Cell matrix in upper triangular column vector form.

    Returns:
       A list containing the lattice vector lengths and the angles between them.
    """

    a = float(h[0, 0])
    b = math.sqrt(h[0, 1]**2 + h[1, 1]**2)
    c = math.sqrt(h[0, 2]**2 + h[1, 2]**2 + h[2, 2]**2)
    gamma = math.acos(h[0, 1] / b)
    beta = math.acos(h[0, 2] / c)
    alpha = math.acos(np.dot(h[:, 1], h[:, 2]) / (b * c))

    return a, b, c, alpha, beta, gamma


def genh2abc(h):
    """ Returns a description of the cell in terms of the length of the
       lattice vectors and the angles between them in radians.

    Takes the representation of the system box in terms of a full matrix
    of row vectors, and returns the representation in terms of the
    lattice vector lengths and the angles between them in radians.

    Args:
       h: Cell matrix in upper triangular column vector form.

    Returns:
       A list containing the lattice vector lengths and the angles between them.
    """

    a = math.sqrt(np.dot(h[0], h[0]))
    b = math.sqrt(np.dot(h[1], h[1]))
    c = math.sqrt(np.dot(h[2], h[2]))
    gamma = math.acos(np.dot(h[0], h[1]) / (a * b))
    beta = math.acos(np.dot(h[0], h[2]) / (a * c))
    alpha = math.acos(np.dot(h[2], h[1]) / (b * c))

    return a, b, c, alpha, beta, gamma


def h2abc_deg(h):
    """Returns a description of the cell in terms of the length of the
       lattice vectors and the angles between them in degrees.

    Takes the representation of the system box in terms of an upper triangular
    matrix of column vectors, and returns the representation in terms of the
    lattice vector lengths and the angles between them in degrees.

    Args:
       h: Cell matrix in upper triangular column vector form.

    Returns:
       A list containing the lattice vector lengths and the angles between them
       in degrees.
    """

    (a, b, c, alpha, beta, gamma) = h2abc(h)
    return a, b, c, alpha * 180 / math.pi, beta * 180 / math.pi, gamma * 180 / math.pi


def abc2h(a, b, c, alpha, beta, gamma):
    """Returns a lattice vector matrix given a description in terms of the
    lattice vector lengths and the angles in between.

    Args:
       a: First cell vector length.
       b: Second cell vector length.
       c: Third cell vector length.
       alpha: Angle between sides b and c in radians.
       beta: Angle between sides a and c in radians.
       gamma: Angle between sides b and a in radians.

    Returns:
       An array giving the lattice vector matrix in upper triangular form.
    """

    h = np.zeros((3, 3), float)
    h[0, 0] = a
    h[0, 1] = b * math.cos(gamma)
    h[0, 2] = c * math.cos(beta)
    h[1, 1] = b * math.sin(gamma)
    h[1, 2] = (b * c * math.cos(alpha) - h[0, 1] * h[0, 2]) / h[1, 1]
    h[2, 2] = math.sqrt(c**2 - h[0, 2]**2 - h[1, 2]**2)
    return h
