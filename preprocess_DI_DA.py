#!/usr/bin/env python
# coding: utf-8
import xarray as xr
import numpy as np
import warnings
import tqdm
import preprocess_amm7_functions as prep

if __name__=="__main__":

    #warnings.filterwarnings("ignore")

    from dask.distributed import Client
    client = Client()
    print(client.dashboard_link)
    
    # Note - which year/years - which depth - surface, bottom, depth integrated
    # Not actually used here
    year_beg = 2000
    year_end = 2004

    depth = 200
    classification = 'physics'
   
    # Variables to train on
   
    vars = prep.cluster_vars(classification)

    print('Loading data')
    #Load Data, combine some variables
    xsl = slice(15,-15)
    ysl = slice(15,-15)

    if depth == 'surface':
        input_path = '/data/thaumus2/scratch/hpo/COMFORT/baseline_archerfull/200[0-4]/**/'
        full_filenames = input_path+'amm7_1d_*_ptrc_T.nc'
    elif classification == 'biogeo':
        input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
        full_filenames = input_path+'amm7_mean_2000-2004_all_depths_biogeo.nc'
    elif classification == 'ecosys':
        input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
        full_filenames = input_path+'amm7_mean_2000-2004_all_depths_ecosys_full.nc'
    elif classification == 'physics':
        input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
        full_filenames = input_path+'amm7_mean_2000-2004_all_depths_physics.nc'

    grd = xr.open_dataset('/data/sthenno1/to_archive/yuti/yuti-SSB-AMM7-hindcasts/mesh_mask.nc').isel(t=0)

    grd = prep.create_mask(grd,depth)

    for var_name in vars:
        # need to do each variable individually as otherwise uses too much memory
        if classification in ['ecosys','biogeo','benthic','ecosys_full']:
            if depth == 'DI':
                if var_name == 'Phytoplankton':
                    vars_to_open = ['P1_c', 'P2_c', 'P3_c', 'P4_c']
                elif var_name == 'Zooplankton':
                    vars_to_open = ['Z4_c', 'Z5_c', 'Z6_c']
                elif var_name == 'DOC':
                    vars_to_open = ['R1_c', 'R2_c', 'R3_c']
                elif var_name == 'POC':
                    vars_to_open = ['R4_c', 'R6_c', 'R8_c']
                else:
                    vars_to_open =  [var_name] #+ ['e3t']
                print(vars_to_open)
                ds = xr.open_mfdataset(full_filenames,chunks={'deptht':51,'x':100,'y':100},data_vars=vars_to_open)
                print(ds)
            elif type(depth) == int:
                ds = xr.open_mfdataset(full_filenames,chunks={'deptht':51,'x':100,'y':100},data_vars=[var_name])
                print(ds,flush=True)
            else:
                # this will mean the loop does the same thing multiple times - don't use
                ds = xr.open_mfdataset(full_filenames,data_vars=vars)

        if classification == 'ecosys':
            if var_name == 'Phytoplankton':
                ds['Phytoplankton'] = ds[['P1_c', 'P2_c', 'P3_c', 'P4_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
            if var_name == 'Zooplankton':
                ds['Zooplankton'] = ds[['Z4_c', 'Z5_c', 'Z6_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
            if var_name == 'DOC':
                ds['DOC'] = ds[['R1_c', 'R2_c', 'R3_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
            if var_name == 'POC':
                ds['POC'] = ds[['R4_c', 'R6_c', 'R8_c']].to_array(dim='sum').sum(dim='sum', skipna=False)

        ds = ds[var_name]
        print(ds, flush=True)
        
        ds = prep.select_depth(ds,grd,classification,depth)

        ds = ds[var_name]
        
        if type(depth) != int:
            ds = ds.isel(x=xsl,y=ysl)

            ds.name = var_name
        
        print(ds, flush=True)

        if var_name == vars[0]:
            ds_full = ds.copy()
        else:
            ds_full = xr.merge([ds_full,ds])

    if depth != None:
        depth_name = str(depth) + '_'
    else:
        depth_name = ''

    ds_full.to_netcdf('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_'+str(year_beg)+'-'+str(year_end)+'_'+depth_name+classification+'.nc')


