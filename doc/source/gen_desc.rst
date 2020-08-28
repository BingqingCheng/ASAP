How-to: asap gen_desc
=========

``asap gen_desc sub_command`` is the descriptor generation command. 
This is in general the first step to do, no matter if you want to map the dataset,
perform regression or any other analysis. Type ``asap gen_desc --help`` and
``asap gen_desc sub_command --help`` for helper strings.

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

