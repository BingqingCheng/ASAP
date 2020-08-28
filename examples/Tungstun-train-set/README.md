Training set for Boron from
Szlachta, Bartók & Csányi Accuracy and transferability of Gaussian
approximation potential models for tungsten PRB 2014

asap gen_desc -f Tungstun_GAP_6.xyz soap

asap map -f ASAP-desc.xyz -dm '[SOAP*]' skpca

asap kde -f ASAP-lowD-map.xyz -dm '[*]' kde_internal plot_pca

