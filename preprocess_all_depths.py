#!/usr/bin/env python
# coding: utf-8
# subset data into relevant variables across all depths
import xarray as xr
import numpy as np
import warnings
import tqdm

if __name__=="__main__":

    #warnings.filterwarnings("ignore")

    from dask.distributed import Client
    client = Client()
    print(client.dashboard_link)
    
    # Note - which year/years - which depth - surface, bottom, depth integrated
    year_beg = 2000
    year_end = 2004

    classification = 'physics'
   
   # Variables to train on
    
    # benthic
    if classification == 'benthic':
        vars = ['Y2_c','Y3_c','Y4_c','H1_c','H2_c','Q1_c','Q6_c']
    # Dale's original set
    if classification == 'Dale':
        vars = ['B1_c','Zooplankton', 'DOM', 'POM', 'Phytoplankton','N4_n','N5_s','O2_o','O3_c','O3_TA','N1_p','N3_n','votemper','vosaline']
    # ecosystem level
    if classification == 'ecosys_full':
        vars = ['P1_c','P2_c','P3_c','P4_c','Z4_c','Z5_c','Z6_c','R1_c','R2_c','R3_c','R4_c','R6_c','R8_c','B1_c']
    # ecosystem level summed
    if classification == 'ecosys':
        vars = ['Phytoplankton','Zooplankton','DOC','POC','B1_c']
    # biogeochemistry
    if classification == 'biogeo':
        vars = ['N1_p','N3_n','N4_n','N5_s','O2_o','O3_c','O3_TA']
    # physics
    if classification == 'physics':
        vars = ['votemper','vosaline','PAR','mldr10_1']
        vars_grid = ['votemper','vosaline','qsr','mldr10_1']
        vars_grid2 = ['votemper','vosaline','mldr10_1']
        vars_ptrc = ['light_xEPS']

    print('Loading data')
    #Load Data, combine some variables

    input_path = '/data/thaumus2/scratch/hpo/COMFORT/baseline_archerfull/200[0-4]/**/'
    full_filenames = input_path+'amm7_1d_*_ptrc_T.nc'

    if classification == 'physics':
        grd = xr.open_dataset('/data/sthenno1/to_archive/yuti/yuti-SSB-AMM7-hindcasts/mesh_mask.nc').isel(t=0)
        grd = grd.rename_dims({'z':'deptht'})

        ds_grid = xr.open_mfdataset(input_path+'amm7_1d_*_grid_T.nc',data_vars=vars_grid).rename_dims({'y_grid_T':'y','x_grid_T':'x'}).rename({'nav_lat_grid_T':'nav_lat','nav_lon_grid_T':'nav_lon'})
        
        ds_ptrc = xr.open_mfdataset(full_filenames,data_vars=vars_ptrc,chunks={'deptht':51,'x':50,'y':50})
        ds_ptrc = ds_ptrc[vars_ptrc]
        ds_ptrc['PAR'] = 0.48*ds_grid['qsr']*np.exp(-(ds_ptrc['light_xEPS']*grd.e3t_0*grd.tmask).cumsum('deptht'))

        ds = xr.merge([ds_grid[vars_grid2], ds_ptrc['PAR']])
    else:    
        ds = xr.open_mfdataset(full_filenames,data_vars=vars)
    
    print(ds)
    
    if classification == 'ecosys':
        ds['Phytoplankton'] = ds[['P1_c', 'P2_c', 'P3_c', 'P4_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['Zooplankton'] = ds[['Z4_c', 'Z5_c', 'Z6_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['DOC'] = ds[['R1_c', 'R2_c', 'R3_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['POC'] = ds[['R4_c', 'R6_c', 'R8_c']].to_array(dim='sum').sum(dim='sum', skipna=False)

    ds = ds[vars]
    print(ds)
    
    ds = ds.mean('time_counter')
    
    ds.to_netcdf('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_'+str(year_beg)+'-'+str(year_end)+'_all_depths_'+classification+'.nc')


