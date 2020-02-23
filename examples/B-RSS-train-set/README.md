GAP training set for boron.
Deringer, Pickard & Csányi Data-Driven Learning of Total and Local
Energies in Elemental Boron PRL 2018

#
* step 0
frame_select.py -fxyz sps_all.xyz --n 1000 --prefix subset --algo random

* step 1
gen_soap_kmat.py -fxyz subset-random-n-1000.xyz --prefix B --rcut 4 --n 4 --l 4 --g 0.5 --periodic True --plot True

* step 2
kpca.py -fmat B-n4-l4-c4.0-g0.5.kmat -fxyz subset-random-n-1000.xyz -colors energy --prefix B-GAP-train-set --output xyz
