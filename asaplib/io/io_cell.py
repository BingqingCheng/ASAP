"""
Functions for converting between different simulation cell formats
"""

import math

import numpy as np


def pbcdist(q1, q2, h, ih):
    s = np.dot(ih, q1 - q2)
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
    b = math.sqrt(h[0, 1] ** 2 + h[1, 1] ** 2)
    c = math.sqrt(h[0, 2] ** 2 + h[1, 2] ** 2 + h[2, 2] ** 2)
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
    h[2, 2] = math.sqrt(c ** 2 - h[0, 2] ** 2 - h[1, 2] ** 2)
    return h
