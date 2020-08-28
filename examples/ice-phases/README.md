A data set of 15,869 distinct PBE-DFT geometry-optimised ice structures obtained from the paper
"Mapping uncharted territory in ice from zeolite networks to ice structures" 
Nature Communicationsvolume 9, Article number: 2173 (2018)

asap gen_desc -f ice-dataset.xyz soap

asap map -f ASAP-desc.xyz -dm '[*]' -c ice-properties.dat -ccol 3 --clab "energy[meV/H$_2$O]" pc

