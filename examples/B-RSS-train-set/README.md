GAP training set for boron.
Deringer, Pickard & Csányi Data-Driven Learning of Total and Local
Energies in Elemental Boron PRL 2018

asap gen_desc -f sps_all.xyz -p B-RSS soap

asap map -f B-RSS.xyz -dm '[*]' -c energy -nbs pca

asap fit -f B-RSS.xyz -dm '[*]' -y energy -nbs ridge

