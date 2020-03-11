"""
TestCase object to be used by the testing framework of the library

"""

import unittest

import numpy as np


class ASAPlibTestCase(unittest.TestCase):
    """
    Mainly taken from quippy's test case object, also mainly written by Tamas K. Stenczel
    See Also
    --------
    https://github.com/libAtoms/quip
    """

    def assertEqual(self, first, second, msg=None):
        if first == second:
            return
        self.fail('{} != {}'.format(first, second))

    def assertArrayAlmostEqual(self, first, second, tol=1e-7):
        first = np.array(first)
        second = np.array(second)
        self.assertEqual(first.shape, second.shape)

        if np.isnan(first).any():
            self.fail('Not a number (NaN) found in first array')
        if np.isnan(second).any():
            self.fail('Not a number (NaN) found in second array')

        absdiff = abs(first - second)
        if np.max(absdiff) > tol:
            print('First array:\n{}'.format(first))
            print('\n \n Second array:\n{}'.format(second))
            print('\n \n Abs Difference:\n{}'.format(absdiff))
            self.fail('Maximum abs difference between array elements is {} at location {}'.format(np.max(absdiff),
                                                                                                  np.argmax(absdiff)))

    def assertArrayIntEqual(self, first, second):
        first = np.array(first)
        second = np.array(second)
        self.assertEqual(first.shape, second.shape)

        if not np.all(first == second):
            print('First array:\n{}'.format(first))
            print('\n \n Second array:\n{}'.format(second))
            self.fail('Two bool arrays not matching, difference is\n {}'.format(first == second))
