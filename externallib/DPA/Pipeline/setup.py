#! /usr/bin/env python
"""Density Peak Advanced clustering algorithm, scikit-learn compatible."""

import codecs
import os
import numpy

from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize

debug = False
numpy_include_dir = numpy.get_include()


EXTENSIONS = [
    Extension('_DPA', ['_DPA.pyx'],
        include_dirs=[numpy_include_dir,],
        define_macros=[('CYTHON_TRACE', '1' if debug else '0')],
        extra_compile_args=['-O3'],
        ),
    Extension('_PAk', ['_PAk.pyx'],
        include_dirs=[numpy_include_dir,],
        define_macros=[('CYTHON_TRACE', '1' if debug else '0')],
        extra_compile_args=['-O3'],
        )
]


setup(ext_modules = cythonize(EXTENSIONS),
      include_dirs=[numpy.get_include()])
