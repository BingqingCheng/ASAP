A data set of small molecules selected from QM7b data set, with dft formation energy. 

#
* step 1: use two sets of SOAP descriptors & print out atomic soap

gen_soap_descriptors.py -fxyz small_molecules-1000.xyz -param_path two_soap_param --peratom true

* step 2: visualize

pca.py -fxyz ASAP-soapparam-two_soap_param.xyz -fmat SOAPPARAM-two_soap_param -colors dft_formation_energy_per_atom_in_eV

* step 3: predict

ridge_regression.py -fxyz ASAP-soapparam-two_soap_param.xyz -fmat SOAPPARAM-two_soap_param -fy dft_formation_energy_per_atom_in_eV
