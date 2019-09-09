This is a data set of 64 water molecules that is used as the training set of a neural network potential of water reported in
Cheng, Engel, Behler, Dellago & Ceriotti Ab initio thermodynamics of liquid
and solid water PNAS 2019

#
* step 1
../../asap/gen_soap_kmat.py -fxyz dataset_1593_eVAng.xyz --prefix H2O --rcut 4.0 --n 4 --l 6 --g 0.3 --periodic True --plot True

* step 2
../../asap/kernel_density_estimation.py -kmat H2O-n4-l6-c4.0-g0.3.kmat --pc1 0 --pc2 1

* step 3
../../asap/clustering.py -kmat H2O-n4-l6-c4.0-g0.3.kmat
