#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import warnings
import tqdm
import preprocess_amm7_functions as prep
import nctoolkit as nc

if __name__=="__main__":

    #warnings.filterwarnings("ignore")

    from dask.distributed import Client
    client = Client()
    print(client.dashboard_link)
    
    # Note - which year/years - which depth - surface, bottom, depth integrated
    year_beg = 2000
    year_end = 2004

    # for this script, should be an number - integer or float
    depth = 10
    classification = 'biogeo'
   
    # Variables to train on
    vars = prep.cluster_vars(classification)

    print('Loading data')
    #Load Data, combine some variables
    xsl = slice(15,-15)
    ysl = slice(15,-15)

    if classification == 'biogeo':
        input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
        full_filenames = input_path+'amm7_mean_2000-2004_all_depths_biogeo.nc'
    elif classification == 'ecosys':
        input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
        full_filenames = input_path+'amm7_mean_2000-2004_all_depths_ecosys_full.nc'

    grd = xr.open_dataset('/data/sthenno1/to_archive/yuti/yuti-SSB-AMM7-hindcasts/mesh_mask.nc').isel(t=0)

    grd = prep.create_mask(grd,depth)

    ds = xr.open_mfdataset(full_filenames) #,data_vars=vars)
    print(ds)

    if classification == 'ecosys':
        ds['Phytoplankton'] = ds[['P1_c', 'P2_c', 'P3_c', 'P4_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['Zooplankton'] = ds[['Z4_c', 'Z5_c', 'Z6_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['DOC'] = ds[['R1_c', 'R2_c', 'R3_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['POC'] = ds[['R4_c', 'R6_c', 'R8_c']].to_array(dim='sum').sum(dim='sum', skipna=False)

    ds = ds[vars]
    print(ds)

    ds = prep.select_depth(ds,grd,classification,depth)

    ds = ds.isel(x=xsl,y=ysl)
    
    if depth != None:
        depth_name = depth + '_'
    else:
        depth_name = ''

    ds.to_netcdf('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_'+str(year_beg)+'-'+str(year_end)+'_'+depth_name+classification+'.nc')


