A data set of small molecules, with bandgaps

#
* step 1: use two sets of SOAP descriptors & print out atomic soap
gen_soap_descriptors.py -fxyz small_molecules-1000.xyz -multisoap two_soap_param --peratom true
