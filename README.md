<p align="left">
  <img src="ASAP-logo.png" width="500" title="logo">
</p>

# ASAP 
Automatic Selection And Prediction tools for materials and molecules

#

Type asap and use the sub-commands for various tasks.

e.g. in the folder ./tests/

to generate SOAP descriptors:

`asap gen_desc --fxyz small_molecules-1000.xyz soap`

for columb matrix:

`asap gen_desc -f small_molecules-1000.xyz --no-periodic cm`

for pca map:

`asap map -f small_molecules-SOAP.xyz -dm [SOAP-n4-l3-c1.9-g0.23] -c dft_formation_energy_per_atom_in_eV pca`

#
python 3

Installation:

python3 setup.py install --user

Requirements:

+ numpy scipy scikit-learn json ase dscribe umap-learn PyYAML click rdkit pytest

Add-Ons:
+ (for finding symmetries of crystals) spglib 
+ (for annotation without overlaps) adjustText
+ The FCHL19 representation requires code from the development brach of the QML package. Instructions on how to install the QML package can be found on https://www.qmlcode.org/installation.html.

One can use the following comments for installing the packages:

pip3 install --upgrade pip

python3 -m pip install --user somepackage    .or.    pip3 install --user somepackage

#
In the directory ./scripts/ and ./tools/ you can find a selection of other python tools:


* frame_select.py: select a subset of the xyz frames based on random or farthest point sampling selection.

* kernel_density_estimate.py: does principle component analysis on the kernel matrix, estimates kernel density, and makes plots

* clustering.py: does clustering tasks based on the kernel matrix, does plotting as well.

