this is an example on a single frame of a Lennard-Jones system. The system contains a solid nucleus inside undercooled liquid. The file "FCCUBIC.list" provides a list of FCCUBIC order parameter, which distinguishes atomic enviroments in a fcc-like environment, of each atom in the frame.
 
we use the ASAP tools to identify the difference in atomic environments of the atoms inside this system.

#
* step 1
../../asap/gen_soap_descriptors.py -fxyz lj-nuclei.xyz --rcut 4 --n 4 --l 6 --g 0.2 --periodic True

* step 2
../../asap/pca.py -fmat ASAP-n4-l6-c4.0-g0.2.desc -colors FCCUBIC.list
