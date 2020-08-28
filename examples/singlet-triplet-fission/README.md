This is a data set that provides the band gaps and some other informations of organic molecules.
we try to use the asap tools to visualize the data and to predict properties
https://github.com/SMTG-UCL/singlet-fission-screening

asap gen_desc -f all.xyz --no-periodic soap

asap fit -f ASAP-desc.xyz -dm '[SOAP*]' -y td_triplet -lc 10 ridge
