How-to: asap gen_desc
=========

``asap gen_desc sub_command`` is the descriptor generation command. 
This is in general the first step to do, no matter if you want to map the dataset,
perform regression or any other analysis. Type ``asap gen_desc --help`` and
``asap gen_desc sub_command --help`` for helper strings.

First part of the command ``asap gen_desc``
***************

The first part ``asap gen_desc`` will be evaluated before the
descriptor specific ones, we setup the general stuff here, such as read
the files.

Table of specific options:

  -s, --stride INTEGER            Read in the xyz trajectory with X stide.
                                  Default: read/compute all frames.

  --periodic / --no-periodic      Is the system periodic? If not specified,
                                  will infer from the XYZ file.

  -i, --in_file, --in PATH        The state file that includes a dictionary-
                                  like specifications of descriptors to use.

  -f, --fxyz TEXT                 Input file that contains XYZ coordinates.
                                  See a list of possible input formats:
                                  https://wiki.fysik.dtu.dk/ase/ase/io/io.html
                                  If a wildcard * is used, all files matching
                                  the pattern is read.

  --fxyz_format TEXT              Additional info for the input file format.
                                  e.g.
                                  {"format":"lammps-
                                  data","units":"metal","style":"full"}

  -p, --prefix TEXT               Prefix to be used for the output file.

  -np, --number_processes,        Number of processes when compute the
  --nprocess INTEGER              descriptors in parrallel.  [default: 1]

  --help                          Show this message and exit.


sub-commands
*************

sub-commands that controls the actual generation of the descriptor matrix:

======  =======================================
option  description 
======  =======================================
  acsf  Generate ACSF descriptors
  cm    Generate the Coulomb Matrix descriptors
  run   Running analysis using input files
  soap  Generate SOAP descriptors
======  =======================================

soap
***********

  Generate SOAP descriptors

Options:
  -c, --cutoff FLOAT              Cutoff radius
  -n, --nmax INTEGER              Maximum radial label
  -l, --lmax INTEGER              Maximum angular label (<= 9)
  --rbf [gto|polynomial]          Radial basis function  [default: gto]
  -sigma, -g, --atom-gaussian-width FLOAT
                                  The width of the Gaussian centered on atoms.
                                  [default: 0.5]

  --crossover / --no-crossover    If to included the crossover of atomic
                                  types.  [default: False]

  -u, --universal_soap, --usoap [none|smart|minimal|longrange]
                                  Try out our universal SOAP parameters.
                                  [default: minimal]

  --tag TEXT                      Tag for the descriptors.
  -pa, --peratom                  Save the per-atom local descriptors.
                                  [default: False]

  -e, --element_wise              element-wise operation to get global
                                  descriptors from the atomic soap vectors
                                  [default: False]

  -z, --zeta INTEGER              Moments to take when converting atomic
                                  descriptors to global ones.

  -r, --reducer_type TEXT         type of operations to get global descriptors
                                  from the atomic soap vectors, e.g.
                                  [average], [sum], [moment_avg],
                                  [moment_sum].  [default: average]

  --help                          Show this message and exit.


acsf
**********

cm
**********

run
**********


.. note::  More documentation to be added. 

