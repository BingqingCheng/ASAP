<p align="left">
  <img src="ASAP-logo.png" width="500" title="logo">
</p>

# ASAP 
Automatic Selection And Prediction tools for materials and molecules

# Basic usage

Type asap and use the sub-commands for various tasks.

e.g. in the folder ./tests/

to generate SOAP descriptors:

`asap gen_desc --fxyz small_molecules-1000.xyz soap`

for columb matrix:

`asap gen_desc -f small_molecules-1000.xyz --no-periodic cm`

for pca map:

`asap map -f small_molecules-SOAP.xyz -dm [SOAP-n4-l3-c1.9-g0.23] -c dft_formation_energy_per_atom_in_eV pca`

# Installation & requirements

python 3

Installation:

python3 setup.py install --user

Requirements:

+ numpy scipy scikit-learn json ase dscribe umap-learn PyYAML click

Add-Ons:
+ (for finding symmetries of crystals) spglib 
+ (for annotation without overlaps) adjustText
+ The FCHL19 representation requires code from the development brach of the QML package. Instructions on how to install the QML package can be found on https://www.qmlcode.org/installation.html.

One can use the following comments for installing the packages:

pip3 install --upgrade pip

python3 -m pip install --user somepackage    .or.    pip3 install --user somepackage

# How to add your own atomic or global descriptors

* To add a new atomic descriptor, add a new `Atomic_Descriptor` class in the asaplib/descriptors/atomic_descriptors.py. As long as it has a `__init__()` and a `create()` method, it should be competitable with the ASAP code. The `create()` method takes an ASE Atoms object as input (see: [ASE](https://wiki.fysik.dtu.dk/ase/ase/atoms.html))

* To add a new global descriptor, add a new `Global_Descriptor` class in the asaplib/descriptors/global_descriptors.py. As long as it has a `__init__()` and a `create()` method, it is fine. The `create()` method also takes the Atoms object as input.

# Additional tools
In the directory ./scripts/ and ./tools/ you can find a selection of other python tools:


* frame_select.py: select a subset of the xyz frames based on random or farthest point sampling selection.

* kernel_density_estimate.py: does principle component analysis on the kernel matrix, estimates kernel density, and makes plots

* clustering.py: does clustering tasks based on the kernel matrix, does plotting as well.

