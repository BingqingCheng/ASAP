../../asap/gen_kmat.py -fxyz rss.xyz -fdict knownphases.xyz --prefix test --rcut 4 --n 6 --l 6 --g 0.5 --periodic True --plot True

../../asap/kpca.py -kmat test-n6-l6-c4.0-g0.5.kmat -tags knowntags.dat --prefix test --d 10 --pc1 0 --pc2 1

../../asap/clustering.py -kmat test-n6-l6-c4.0-g0.5.kmat -tags knowntags.dat --prefix test
