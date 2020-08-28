this is an example on a single frame of a nanoparticle. 
we use the ASAP tools to identify the difference in atomic environments of the atoms inside this nanoparticle.

asap gen_desc -f Ni-icosa.xyz soap --peratom

asap cluster -f ASAP-desc.xyz -dm '[*]' -ua fdb plot_pca
