Training set for Boron from
Szlachta, Bartók & Csányi Accuracy and transferability of Gaussian
approximation potential models for tungsten PRB 2014

* step 0
../../asap/frame_select.py -fxyz Tungstun_GAP_6.xyz --n 1000 --prefix subset --algo random

* step 1
../../asap/gen_soap_kmat.py -fxyz subset-random-n-1000.xyz --prefix W --rcut 5 --n 4 --l 6 --g 0.5 --periodic True

* step 2
../../asap/kpca.py -kmat W-n4-l6-c5.0-g0.5.kmat -fxyz subset-random-n-1000.xyz -colors energy --prefix W-GAP-train-set

* step 3
../../asap/kpca.py -kmat W-n4-l6-c5.0-g0.5.kmat -fxyz subset-random-n-1000.xyz -colors volume --prefix W-GAP-train-set
