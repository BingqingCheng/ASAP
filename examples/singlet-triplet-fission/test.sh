../../asap/gen_kmat.py -fxyz all.xyz --prefix test --rcut 4 --n 6 --l 6 --g 0.5 --periodic False --plot True

../../asap/kpca_4_kmat.py -kmat test-n6-l6-c4.0-g0.5.kmat --prefix test --d 10 --pc1 0 --pc2 1
