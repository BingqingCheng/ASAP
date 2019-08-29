../../asap/gen_kmat.py -fxyz subset.xyz --prefix subset --rcut 2.5 --n 4 --l 6 --g 0.4 --periodic False --plot True

../../asap/kpca.py -kmat subset-n4-l6-c2.5-g0.4.kmat --prefix subset --d 10 --pc1 0 --pc2 1
