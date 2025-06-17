# ModelViz

Tool to classify marine biogeochemical output from numerical models

Written by rmi, dapa & dmof

## Preprocessing
*preprocess_amm7_functions.py* <br>
Functions needed to run different preprocessing scripts.<br>
*preprocess_all_depths.py* <br>
First script to run. Extracts relevant variables and takes temporal mean for physical,biogeochemical and ecological variables. For physical - calculates PAR from qsr.<br>
*preprocess_amm7_mean.py* <br>
Use for surface biogeochemical and ecological sets (faster)<br>
*preprocess_DI_DA.py*<br>
Use for depth integrated, depth averaged and bottom biogeochemical and ecological sets. Can use for surface but slower.<br>
*preprocess_amm7_mean_one_depth.py*<br>
Extracts data at specified depth (numeric). Works for biogeochemical and ecological variables.<br>
*preprocess_physics.py*<br>
Takes all_depths_physics and calculates physics data at different depths.

## Metrics
*silhouette_nvars.py*<br>
Calculates silhouette score for inputs with different numbers of variables and clusters<br>
*rand_index.py*<br>
*rand_index_depth.py*<br>
*remove_one_var.py*<br>
Calculates rand index between cluster sets with one variable removed and original set

## Clustering
*Modelviz.py*<br>
Contains functions for applying clustering to data

## Plotting
*kmeans-paper-plots.ipynb*<br>
Produces figure 4<br>
*kmeans-paper-plots-illustrate-normalisation.ipynb*<br>
Produces figure 2<br>
*kmeans-paper-plots-depths.ipynb*<br>
Produces figures 5-7<br>
*plot_silhouette.ipynb*<br>
Produces figure 3
