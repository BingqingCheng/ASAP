.. asapdoc documentation master file, created by
   sphinx-quickstart on Mon Aug  3 19:06:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Quick & basic example
===================================

Step 1: generate a design matrix
----------------------------------

The first step for a machine-learning analysis or visualization is to generate a "design matrix" made from either global descriptors or atomic descriptors. To do this, we supply ``asap gen_desc`` with an input file that contains the atomic coordintes. Many formats are supported; anything can be read using [ase.io](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) is supported. You can use a wildcard to specify the list of input files that matches the pattern (e.g. ``POSCAR*``, ``H*``, or ``*.cif``). However, it is most robust if you use an extended xyz file format (units in angstrom, additional info and cell size in the comment line).

As a quick example, in the folder ./tests/

to generate SOAP descriptors:

.. code-block:: sh

    asap gen_desc --fxyz small_molecules-1000.xyz soap

for columb matrix:

.. code-block:: sh

    asap gen_desc -f small_molecules-1000.xyz --no-periodic cm

Step 2: generate a low-dimensional map
--------------------------------------

After generating the descriptors, one can make a two-dimensional map (``asap map``), or regression model (``asap fit``), or clustering (``asap cluster``), or select a subset of frames (``asap select``), or do a clustering analysis (``asap cluster``), or estimate the probablity of observing each sample (``asap kde``).

For instance, to make a pca map:

.. code-block:: sh

    asap map -f small_molecules-SOAP.xyz -dm '[SOAP-n4-l3-c1.9-g0.23]' -c dft_formation_energy_per_atom_in_eV pca

You can specify a list of descriptor vectors to include in the design matrix, e.g. ``'[SOAP-n4-l3-c1.9-g0.23, SOAP-n8-l3-c5.0-g0.3]'``

one can use a wildcard to specify the name of all the descriptors to use for the design matrix, e.g.

.. code-block:: sh

    asap map -f small_molecules-SOAP.xyz -dm '[SOAP*]' -c dft_formation_energy_per_atom_in_eV pca

or even

.. code-block:: sh

    asap map -f small_molecules-SOAP.xyz -dm '[*]' -c dft_formation_energy_per_atom_in_eV pca

Step 2+: interactive visualization
----------------------------------

Using ``asap map``, a png figure is generated. In addition, the code also output the low-dimensional coordinates of the structures and/or atomic environments. The default output is extended xyz file. One can also specify a different output format using ``--output`` or ``-o`` flag. and the available options are ``xyz``, ``matrix`` and ``chemiscope``. 

* If one select ``chemiscope`` format, a ``*.json.gz`` file will be writen, which can be directly used as the input of [chemiscope](https://github.com/cosmo-epfl/chemiscope)

* If the output is in ``xyz`` format, it can be visualized interactively using [projection_viewer](https://github.com/chkunkel/projection_viewer).

