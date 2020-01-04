A data set of small molecules selected from QM7b data set, with band gaps. 

#
* step 1: use two sets of SOAP descriptors & print out atomic soap
gen_soap_descriptors.py -fxyz small_molecules-1000.xyz -multisoap two_soap_param --peratom true

* step 2: visualize
pca.py -fxyz ASAP-multisoap-two_soap_param.xyz -fmat MULTISOAP-two_soap_param -colors dft_formation_energy_per_atom_in_eV

* step 3: predict
ridge_regression.py -fxyz ASAP-multisoap-two_soap_param.xyz -fmat MULTISOAP-two_soap_param -fy dft_formation_energy_per_atom_in_eV
