this is an example on a single frame of a nanoparticle. 
we use the ASAP tools to identify the difference in atomic environments of the atoms inside this nanoparticle.
It is assumed that you are in the ASAP/examples/Ni-icosahedral-np directory.

#
* step 1
gen_soap_kmat.py -fxyz Ni-icosa.xyz --prefix Ni-icosa --rcut 4 --n 6 --l 6 --g 0.5 --periodic True --plot True

* step 2.1
kpca.py -fmat Ni-icosa-n6-l6-c4.0-g0.5.kmat --prefix Ni-icosa --d 10 --pc1 0 --pc2 1

* step 2.2
kernel_density_estimation.py -fmat Ni-icosa-kpca-d10.coord --prefix Ni-icosa --pc1 0 --pc2 1

* step 3
clustering.py -kmat Ni-icosa-n6-l6-c4.0-g0.5.kmat --prefix Ni-icosa
