# CSIRO

## Kaggle competition

Not really in for the money or the fame, but this one have something ...
well, smell like Alberta I guess
Lets have fun
and scats

## Ressources
https://www.kaggle.com/competitions/csiro-biomass/overview

## Installation
Need to install the special lib from sheng & menard
'''
git clone https://github.com/SihaoCheng/scattering_transform/ 
cp -r scattering_transform/scattering .
'''


## Notes
pour l'embedding j'ai un embedding par patch, est ce que je joue là dessus en supossant une distribution uniforme de la biomasse sur les images ? (et du coup un patch est associé à la biomass totale présente sur l'image / nb de patch)
ou est ce que c'est une connerie et à ce moment là je fusionne les embeddings ? sauf que là du coup je me retrouve avec une representation du bazar en très grande dimension
ou je fais juste un patch de la taille de l'image, pour commencer ça peut être simple -> c'est ce que j'ai fais pour commencer

Une image a plusieurs labels associé, en fait j'ai l'impression que la prediction doit etre la quantitée de biomass par espece presente, pas la quantite totale
donc la generation de dataset est devenue d'un coup plus tricky, il faut que je reflechisse un peu -> la col target_name est la clée, plusieurs target par image,
on peut filtrer là dessus et faire plusieurs dataset

