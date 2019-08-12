# step 1
../../asap/gen_kmat.py -fxyz Ni-icosa.xyz --prefix Ni-icosa --rcut 4 --n 6 --l 6 --g 0.5 --periodic True --plot True

# step 2
../../asap/kpca_4_kmat.py -kmat Ni-icosa-n6-l6-c4.0-g0.5.kmat --prefix Ni-icosa --d 10 --pc1 0 --pc2 1
