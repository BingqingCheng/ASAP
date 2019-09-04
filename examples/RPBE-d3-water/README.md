Training set for RPBE+D3 water/ice
Morawietz, et al. How van der Waals interactions determine the unique
properties of water PNAS 2016

#

* step 0
frame_select.py -fxyz RPBE-d3-water.xyz --n 2000 --prefix RPBE-d3-water

* step 1
gen_soap_kmat.py -fxyz RPBE-d3-water-random-n-2000.xyz --prefix H2O --rcut 4.0 --n 4 --l 6 --g 0.3 --periodic True --plot True

* step 2
kpca.py -kmat H2O-n4-l6-c4.0-g0.3.kmat -fxyz RPBE-d3-water-random-n-2000.xyz -colors energy --prefix H2O

* step 3
kpca.py -kmat H2O-n4-l6-c4.0-g0.3.kmat -fxyz RPBE-d3-water-n-1000.xyz -colors volume --prefix H2O
