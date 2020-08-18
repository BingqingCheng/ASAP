"""
TODO: Module-level description
"""

import glob

import ase.build.supercells as sc
from ase.io import read


def ciftoxyzfunc(name, size=[1, 1, 1]):
    a = read(name)
    b = sc.make_supercell(a, [[size[0], 0, 0], [0, size[1], 0], [0, 0, size[2]]])
    b.write(name + '.xyz', format='xyz')


ciflist = glob.glob("*.cif")
for c in ciflist:
    ciftoxyzfunc(c)
