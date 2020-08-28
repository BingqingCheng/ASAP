How-to: asap gen_desc
=========

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

.. note:: when glob pattern is passed it should be quote: e.g. 
    ``asap gen_desc -f *.cell`` will not work 
    but ``asap gen_desc -f "*.cell"`` does.

.. note:: if the program gets a bit confused about the file format, try supply some additional information using ``--fxyz_format '{...}'`` flag.

Output
-------

The code will output two files

* ``${prefix}-desc.xyz``, where ``${prefix}`` is determined by the string that follows ``--prefix`` flag (default: ASAP). This file is an extended xyz file that contain the design matrix (descriptors for each structure and/or atomic environments).

* ``${prefix}-desc.yaml`` that contains all the meta information about how the design matrix was generated (i.e. which descriptor, hyper-parameter was used.)

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

