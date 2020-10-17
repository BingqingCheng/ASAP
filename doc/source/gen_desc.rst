How-to: asap gen_desc
=====================

``asap gen_desc sub_command`` is the descriptor generation command. 
This is in general the first step to do, no matter if you want to map the dataset,
perform regression or any other analysis. Type ``asap gen_desc --help`` and
``asap gen_desc sub_command --help`` for helper strings.

Input
------

``asap gen_desc`` can read any format that is also supported by ASE.[ase.io](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) is supported. 
For example:
xyz, lammps-data, cif, cell, vasp, res, gromacs, ...
However, it is most thoroughly tested on extended xyz files (units in angstrom, additional info and cell size in the comment line). 
It can read in lots of files using a wildcard and some common pattern e.g. ``H*.cell``. 

.. note:: when glob pattern is passed it should be quoted: e.g. 
    ``asap gen_desc -f *.cell`` will not work 
    but ``asap gen_desc -f "*.cell"`` does.

.. note:: if the program gets a bit confused about the file format, try supply some additional information using ``--fxyz_format '{...}'`` flag.

Output
-------

The code will output two files

* ``${prefix}-desc.xyz``, where ``${prefix}`` is determined by the string that follows ``--prefix`` flag (default: ASAP). This file is an extended xyz file that contain the design matrix (descriptors for each structure and/or atomic environments).

* ``${prefix}-desc.yaml`` that contains all the meta information about how the design matrix was generated (i.e. which descriptor, hyper-parameter was used.)

Methods
-------

There are two types of descriptors for atomic structures: the first type (e.g. ACSF, SOAP) starts from atomic descriptors for each atom in the structure, and then all the atomic descriptors associated with a structure are reduced to a global descriptor. The second type (e.g. Coulumb Matrix) directly generates global descriptors.

To reduce the atomic descriptors of all atoms in a structure to a single vector representing the whole strucuture, ``asap`` calls a ``reducer``, and there are different options. For example, for a structure A, the most straightforward way to get its global descriptor is to simply take the average of the atomic ones, i.e.

.. math::

    \Phi (A) = \dfrac{1}{N_A}\sum_{i \in A}^{N_A} \psi (\mathcal{X}_i).

which is the ``average`` reducer. Alternatively, one can take the ``sum``. The ``moment_average`` or ``moment_average`` reducer first take the moment of the atomic descriptors, e.g.

.. math::

    \Phi (A) = \dfrac{1}{N_A}\sum_{i \in A}^{N_A} \psi^{z} (\mathcal{X}_i).

where z (``--zeta/-z``) is the moment to take when converting atomic descriptors to global ones.

In addition to these, one can perform the summation or the average operation on the per-element basis, by using the flag ``--element_wise/-e``.


Overview of sub-commands
-------------------------

sub-commands that controls the actual generation of the descriptor matrix:

======  =======================================
option  description 
======  =======================================
  acsf  Generate ACSF descriptors
  cm    Generate the Coulomb Matrix descriptors
  run   Running analysis using input files
  soap  Generate SOAP descriptors
======  =======================================


.. click:: asaplib.cli.cmd_asap:gen_desc
   :prog: asap gen_desc
   :nested: full

.. note::  More documentation to be added. 

