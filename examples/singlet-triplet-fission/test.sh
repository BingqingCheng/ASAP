../../asap/gen_kmat.py -fxyz all.xyz --prefix test --rcut 2.5 --n 4 --l 6 --g 0.4 --periodic False --plot True

../../asap/kpca.py -kmat test-n4-l6-c2.5-g0.4.kmat --prefix test --d 10 --pc1 0 --pc2 1
