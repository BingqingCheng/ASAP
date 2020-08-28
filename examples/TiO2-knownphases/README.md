This is a dataset of crystalline structures of TiO2
we use asap tools to do see which stuctures are similar to each other

asap gen_desc -f knownphases.xyz -p known-phases-soap soap

asap map -f known-phases-soap.xyz -dm '[*]' -a knowntags.dat --adjusttext pca
