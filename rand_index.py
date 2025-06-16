import ModelViz
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import glob
import xarray as xr

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
    if classification == 'phys-biogeo':
        return ['votemper','vosaline','mldr10_1','PAR','N1_p','N3_n','O2_o','O3_c','O3_TA','N4_n','N5_s']
    if classification == 'phys-ecosys':
        return ['votemper','vosaline','mldr10_1','PAR','Phytoplankton','Zooplankton','DOC','POC','B1_c']
    if classification == 'biogeo-ecosys':
        return ['N1_p','N3_n','O2_o','O3_c','O3_TA','N4_n','N5_s','Phytoplankton','Zooplankton','DOC','POC','B1_c']
    if classification == 'all':
        return ['votemper','vosaline','mldr10_1','PAR','N1_p','N3_n','O2_o','O3_c','O3_TA','N4_n','N5_s','Phytoplankton','Zooplankton','DOC','POC','B1_c']

def run_initial_model(classification,depth,rep_num):

    train = ModelViz.ModelViz()
    # weird formatting of my data....
    train.x_strip = slice(15,-15)
    train.y_strip = slice(15,-15)

    train.cluster_vars = cluster_var_names(classification)
    if classification == 'biogeo_short':
        classification = 'biogeo'

    if classification == 'benthic':
        filename = glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+classification+'*.nc')
    elif classification == 'phys-biogeo':
        filename = glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_physics.nc')
        filename.extend(glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_biogeo.nc'))
    elif classification == 'phys-ecosys':
        filename = glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_physics.nc')
        filename.extend(glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_ecosys.nc'))
    elif classification == 'biogeo-ecosys':
        filename = glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_biogeo.nc')
        filename.extend(glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_ecosys.nc'))
    elif classification == 'all':
        filename = glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_physics.nc')
        filename.extend(glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_biogeo.nc'))
        filename.extend(glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_ecosys.nc'))
    else:
        filename = glob.glob('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_'+depth+'_'+classification+'.nc')
    
    train.ds = xr.open_mfdataset(filename).drop_dims('axis_nbounds', errors='ignore')
    if 'deptht' in train.ds.dims:
        train.ds = train.ds.squeeze(dim=['deptht'])

    #train.load_mfdata(filename)
    train.load_grid('/data/sthenno1/to_archive/yuti/yuti-SSB-AMM7-hindcasts/mesh_mask.nc')

    #train.summarise_features(sum_vars)
    train.norm = 'stdev'
    train.preprocess(do_slice=False)     
    train.make_tsds()
    
    train.seed = rep_num
    train.n_init = 100
    #train.seed = 810
    train.train(n_clusters=6, save=False,model_name = 'kmeans') #, file_path='model.ks')
    return train

sum_vars={'Phytoplankton': ['P1_c', 'P2_c', 'P3_c', 'P4_c'],
            'Zooplankton': ['Z4_c', 'Z5_c', 'Z6_c'],
            'DOM': ['R1_c', 'R2_c', 'R3_c'],
            'POM': ['R4_c', 'R6_c', 'R8_c']}

total_reps = 10

classification_vec1 = ['ecosys'] #,'biogeo']
depth_vec1 = ['surface'] # ,'surface','surface','bottom','bottom','DA']

classification_vec2 = ['phys-biogeo','phys-ecosys','all']
depth_vec2 = ['surface'] #,'DA','DI','DA','DI','DI']

for i in range(len(classification_vec2)):
    print(classification_vec1[0])
    #print(depth_vec1[i])
    print(classification_vec2[i])
    #print(depth_vec2[i])

    results = np.empty((total_reps+2,1))

    for rep in range(total_reps):
        train1 = run_initial_model(classification_vec1[0],depth_vec1[0],rep)
        train1.predict()
        train2 = run_initial_model(classification_vec2[i],depth_vec2[0],rep)
        train2.predict()

        results[rep] = metrics.rand_score(train1.labels,train2.labels)

    # calculate mean of replicates
    results[-2] = np.mean(results[0:total_reps])
    results[-1] = np.std(results[0:total_reps])
    print('Mean rand index')
    print(results[-2])
    print('Stdev rand index')
    print(results[-1])

#a = 0; b = 0; c = 0; d = 0;
#for i in range(len(labels1)):
#    pos = labels1[i]==labels1
#    a += sum(labels2[i]==labels2[pos])
#    c += sum(labels2[i]!=labels2[pos])
#    pos = labels1[i]!=labels1
#    d += sum(labels2[i]==labels2[pos])
#    b += sum(labels2[i]!=labels2[pos])
#R = (a+b)/(a+b+c+d)
#print(R)
