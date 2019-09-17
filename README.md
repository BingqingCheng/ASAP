<p align="left">
  <img src="ASAP-logo.png" width="500" title="logo">
</p>

# ASAP 
Automatic Selection And Prediction tools for crystal structures

python 3

Requirements:

+ numpy scipy scikit-learn ase dscribe

Add-Ons:
+ spglib adjustText

In the directory ./asap/ you can find a selection of python tools:
* gen_soap_kmat.py: computes kernel matrix between different structures if multiple frames are provides, or the kernel matrix between atomic environments in a structure if only one frame is the input.

* frame_select.py: select a subset of the xyz frames based on random or farthest point sampling selection.

* kpca.py: does principle component analysis on the kernel matrix and makes plots.

* kernel_density_estimate: does principle component analysis on the kernel matrix, estimates kernel density, and makes plots

* clustering.py: does clustering tasks based on the kernel matrix, does plotting well.

* krr.py: quick kernel ridge regression, in order to test if there is enough signal in the data set.

TODOs:
* add a class of methods to compute kernel matrix from basis functions
* add a dendrogram function
