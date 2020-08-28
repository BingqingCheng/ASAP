this is an example on a single frame of a Lennard-Jones system. The system contains a solid nucleus inside undercooled liquid. The file "FCCUBIC.list" provides a list of FCCUBIC order parameter, which distinguishes atomic enviroments in a fcc-like environment, of each atom in the frame.
 
we use the ASAP tools to identify the difference in atomic environments of the atoms inside this system.

asap gen_desc -f 'lj*.xyz' soap -c 3.0 -n 8 -l 6 -g 0.2 -pa

asap map -f ASAP-desc.xyz -dm '[*]' -c FCCUBIC.list -clab 'fcc order parameter' -ua -s journal -o none -p pca pca
