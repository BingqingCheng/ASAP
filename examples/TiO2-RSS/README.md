This is a dataset of crystalline structures of TiO2
we use asap tools to do some analysis and clustering tasks

#
* step 1
../../asap/gen_soap_kmat.py -fxyz rss.xyz -fdict knownphases.xyz --prefix TiO2 --rcut 4 --n 6 --l 6 --g 0.5 --periodic True --plot True

* step 2.1
../../asap/kpca.py -fmat TiO2-n6-l6-c4.0-g0.5.kmat -tags knowntags.dat --prefix TiO2 --d 10 --pc1 0 --pc2 1

* step 2.2
../../asap/kernel_density_estimation.py -fmat TiO2-n6-l6-c4.0-g0.5.kmat -tags knowntags.dat --prefix TiO2

* step 3
../../asap/clustering.py -kmat TiO2-n6-l6-c4.0-g0.5.kmat -tags knowntags.dat --prefix TiO2

