import ModelViz
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

def cluster_var_names(classification):
    if  classification == 'physics':
        return ['votemper','vosaline','mldr10_1','PAR']
    if classification == 'biogeo':
        return ['N1_p','N3_n','O2_o','O3_c','O3_TA','N4_n','N5_s']
    if classification == 'biogeo_short':
        return ['N1_p','N3_n','O2_o','O3_c','O3_TA','N4_n']
    if classification == 'benthic':
        return ['Y2_c','Y3_c','Y4_c','H1_c','H2_c','Q1_c','Q6_c']
    if classification == 'ecosys':
        return ['Phytoplankton','Zooplankton','DOC','POC','B1_c']

def run_initial_model(classification,depth,cluster_vars,run_name,rep_num):
    
    print(cluster_vars)
    
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
    
    train.seed = rep_num
    #train.n_init = 1
    train.train(n_clusters=6, model_name = 'kmeans', save=False)
    train.predict()
    train.get_cluster_info(save=False)
    train.plot_map(savefig=True,file_path='/users/modellers/rmi/Documents/FOCUS/kmeans_figures/one_var/'+run_name+'_map.png')
    return train

sum_vars={'Phytoplankton': ['P1_c', 'P2_c', 'P3_c', 'P4_c'],
            'Zooplankton': ['Z4_c', 'Z5_c', 'Z6_c'],
            'DOM': ['R1_c', 'R2_c', 'R3_c'],
            'POM': ['R4_c', 'R6_c', 'R8_c']}


classification = 'physics'
total_reps = 10

cluster_vars_full = cluster_var_names(classification)

results = np.empty((len(cluster_vars_full),total_reps+2))

for rep in range(total_reps):
    train_orig = run_initial_model(classification,'surface',cluster_vars_full,'original',rep)
    for i in range(len(cluster_vars_full)):
        print(cluster_vars_full[i])
        
        cluster_vars = np.copy(cluster_vars_full)
        cluster_vars = np.delete(cluster_vars,i)
        print(cluster_vars)

        train2 = run_initial_model(classification,'surface',cluster_vars,'remove_'+cluster_vars_full[i],rep)
        #train2.predict()

        results[i,rep] = metrics.rand_score(train_orig.labels,train2.labels)
        print(results[i,rep])

# calculate mean of replicates
for i in range(len(cluster_vars_full)):
    results[i,-2] = np.mean(results[i,0:total_reps])
    results[i,-1] = np.std(results[i,0:total_reps])


np.savetxt('remove_one_var_'+classification+'.txt',results,delimiter=",")
