# ModelViz

Tool to classify marine biogeochemical output from numerical models

Written by rmi, dapa & dmof

## Preprocessing
*preprocess_amm7_functions.py*
Functions needed to run different preprocessing scripts.
*preprocess_all_depths.py*
First script to run. Extracts relevant variables and takes temporal mean for physical,biogeochemical and ecological variables. For physical - calculates PAR from qsr.
*preprocess_amm7_mean.py*
Use for surface biogeochemical and ecological sets (faster)
*preprocess_DI_DA.py*
Use for depth integrated, depth averaged and bottom biogeochemical and ecological sets. Can use for surface but slower.
*preprocess_amm7_mean_one_depth.py*
Extracts data at specified depth (numeric). Works for biogeochemical and ecological variables.
*preprocess_physics.py*
Takes all_depths_physics and calculates physics data at different depths.

## Metrics
*silhouette_nvars.py*
Calculates silhouette score for inputs with different numbers of variables and clusters
*rand_index.py*
*rand_index_depth.py*
*remove_one_var.py*
Calculates rand index between cluster sets with one variable removed and original set

## Clustering
*Modelviz.py*
Contains functions for applying clustering to data

## Plotting
*kmeans-paper-plots.ipynb*
Produces figure 4
*kmeans-paper-plots-illustrate-normalisation.ipynb*
Produces figure 2
*kmeans-paper-plots-depths.ipynb*
Produces figures 5-7
*plot_silhouette.ipynb*
Produces figure 3
