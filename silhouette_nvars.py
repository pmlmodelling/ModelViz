import ModelViz
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from itertools import combinations
import time

def cluster_var_names(classification):
    if  classification == 'physics':
        return ['votemper','vosaline','mldr10_1','qsr']
    if classification == 'biogeo':
        return ['N1_p','N3_n','O2_o','O3_c','O3_TA','N4_n','N5_s']
    if classification == 'biogeo_short':
        return ['N1_p','N3_n','O2_o','O3_c','O3_TA','N4_n']
    if classification == 'benthic':
        return ['Y2_c','Y3_c','Y4_c','H1_c','H2_c','Q1_c','Q6_c']
    if classification == 'ecosys':
        return ['Phytoplankton','Zooplankton','DOC','POC','B1_c']

def load_initial_data(classification,depth,cluster_vars):
    
    print(cluster_vars, flush=True)
    
    train = ModelViz.ModelViz()
    # weird formatting of my data....
    train.x_strip = slice(15,-15)
    train.y_strip = slice(15,-15)

    train.cluster_vars = cluster_vars
    if classification == 'biogeo_short':
        classification = 'biogeo'

    if classification == 'benthic':
        filename = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+classification+'*.nc'
    else:
        filename = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_'+classification+'.nc'

    train.load_mfdata(filename)
    train.load_grid('/data/sthenno1/to_archive/yuti/yuti-SSB-AMM7-hindcasts/mesh_mask.nc')

    #train.summarise_features(sum_vars)
    train.norm = 'stdev'
    train.preprocess(do_slice=False)     
    train.make_tsds()

    return train

sum_vars={'Phytoplankton': ['P1_c', 'P2_c', 'P3_c', 'P4_c'],
            'Zooplankton': ['Z4_c', 'Z5_c', 'Z6_c'],
            'DOM': ['R1_c', 'R2_c', 'R3_c'],
            'POM': ['R4_c', 'R6_c', 'R8_c']}

classification = 'biogeo'

cluster_vars_full = cluster_var_names(classification)

silhouette_vec = np.zeros((19,7,35))
with open('silhouette_5.csv','w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(silhouette_vec.shape))
    for nvar in [5]: #range(1,len(cluster_vars_full)+1):
        # number of variables for run 
        cluster_vars_iter = [list(x) for x in combinations(cluster_vars_full,nvar)]
        print('Number of variables:', flush=True)
        print(nvar, flush=True)

        for niter in range(len(cluster_vars_iter)):
            # number of ways to arrange these variables
            train = load_initial_data(classification,'surface',cluster_vars_iter[niter])
            
            start = time.time()
            for nclust in range(2,21):
                # 2 to 20 clusters 
                print(nclust, flush=True)
                train.train(n_clusters=nclust, save=False,model_name = 'kmeans')
                train.predict()
                silhouette_vec[nclust-2,nvar-1,niter] = metrics.silhouette_score(train.tsds, train.labels,metric='euclidean')
                end = time.time()
                print(end-start, flush=True)
            print('Total time elapsed for iteration:', flush=True)
            print(end-start, flush=True)

        np.savetxt(outfile,silhouette_vec[:,nvar-1,:].squeeze(),delimiter=",")
        outfile.write('# New slice\n')
