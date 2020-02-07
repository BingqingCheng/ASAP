This is a data set that provides the band gaps and some other informations of organic molecules.
we try to use the asap tools to visualize the data and to predict properties
https://github.com/SMTG-UCL/singlet-fission-screening
#

* step 0: select a subset

frame_select.py -fxyz all.xyz -y tda_triplet --n 1000 --algo random

* step 1: generate the kernel matrix

gen_soap_kmat.py -fxyz ASAP-random-n-1000.xyz --rcut 2.5 --n 4 --l 6 --g 0.4 --periodic False --plot True

* step 2: visualize it

kpca.py -fmat ASAP-n4-l6-c2.5-g0.4.kmat --d 10 --pc1 0 --pc2 1 -colors ASAP-random-n-1000-tda_triplet

* step 3: use kernel ridge regression to predict the triplet band gap (tda_triplet)

krr.py -fmat ASAP-n4-l6-c2.5-g0.4.kmat -y ASAP-random-n-1000-tda_triplet --test 0.1 --n 100 --sigma 0.002
