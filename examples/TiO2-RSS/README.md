This is a dataset of crystalline structures of TiO2
we use asap tools to do some analysis and clustering tasks

#
* step 1
python ../../scripts/gen_soap_descriptors.py -fxyz rss.xyz -fdict knownphases.xyz --prefix TiO2 --rcut 4 --n 4 --l 4 --g 0.5

* step 2.1
python ../../scripts/pca.py -fxyz TiO2-n4-l4-c4.0-g0.5.xyz -fmat SOAP-n4-l4-c4.0-g0.5 -tags knowntags.dat --prefix TiO2 --d 10 --pc1 0 --pc2 1

* step 2.2
python ../../scripts/kernel_density_estimation.py -fxyz TiO2-n4-l4-c4.0-g0.5.xyz -fmat TiO2-pca-d10.coord -tags knowntags.dat --prefix TiO2 --d 5

* step 3
python ../../scripts/clustering.py -fmat TiO2-pca-d10.coord -tags knowntags.dat --prefix TiO2

