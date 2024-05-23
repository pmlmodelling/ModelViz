#!/usr/bin/env python
# coding: utf-8

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

    depth = 'bottom'
    classification = 'ecosys'
   
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
        vars = ['votemper','vosaline','mldr10_1']
    #vars = ['Zooplankton', 'fish_c_tot'] # 'fish_pelagic_size_spectrum_slope','
    # Map of names
    names={'votemper':'SST','vosaline':'SSS','N4_n':'Ammonium','N5_s':'Silicon','O2_o':'Oxygen','B1_c':'Bacteria','O3_c':'DIC',\
            'O3_TA':'Alkalinity','Zooplankton':'Zooplankton','DOM':'DOM','POM':'POM','Phytoplankton':'Phytoplankton','N1_p':'Phosphorous','fish_c_tot':'Fish biomass',\
            'fish_pelagic_size_spectrum_slope':'fish_size_spectrum_slope'} #,'N3_n':'Nitrogen',

    print('Loading data')
    #Load Data, combine some variables
    xsl = slice(15,-15)
    ysl = slice(15,-15)

    if depth == 'surface':
        input_path = '/data/thaumus2/scratch/hpo/COMFORT/baseline_archerfull/rundata/200[0-4]/**/'
        full_filenames = input_path+'amm7_1d_*_ptrc_T.nc'
    elif classification == 'biogeo':
        input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
        full_filenames = input_path+'amm7_mean_2000-2004_all_depths_biogeo.nc'
    elif classification == 'ecosys':
        input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
        full_filenames = input_path+'amm7_mean_2000-2004_all_depths_ecosys_full.nc'

    #grd = xr.open_dataset('/data/proteus1/scratch/dapa/AMM7-MONTHLY-SURFACE/domain_cfg.nc').isel(t=0)
    grd = xr.open_dataset('/data/sthenno1/to_archive/yuti/yuti-SSB-AMM7-hindcasts/mesh_mask.nc').isel(t=0)

    #if depth == 'DI':
        #Create 3D mask - not needed for mesh_mask as this is already provided
        # DALE version
        #tmask = 0*grd.e3t_0.values + 1
        #for i in tqdm.tqdm(grd.x):
        #    for j in grd.y:
        #        tmask[int(grd.bottom_level.isel(x=i,y=j).values):,j,i] = 0
        #grd['tmask'] = (('z','y','x'),tmask)

    if depth == 'bottom':
        #Create 3D mask
        # DALE version
        #tmask = 0*grd.e3t_0.values
        #for i in tqdm.tqdm(grd.x):
        #    for j in grd.y:
        #        tmask[int(grd.bottom_level.isel(x=i,y=j).values)-1,j,i] = 1
        # HELEN version (SSB) - is there a different mask which gives the bottom?
        floor = 0*grd.tmask.values
        for i in np.arange(0,floor.shape[1]):
          for j in np.arange(0,floor.shape[2]):
            for k in np.arange(-1,-51,-1):
              if grd.tmask[k,i,j]==1: floor[k,i,j]=1; break
        grd['floor'] = (('z','y','x'),floor)

    # Put mask and depth in grd dataset
    grd = grd.rename_dims({'z':'deptht'})
    
    #if classification in ['ecosys','biogeo','benthic']:
    #    if depth == 'DI':
    #        vars_to_open =  vars + ['e3t']
    #        print(vars_to_open)
    #        ds = xr.open_mfdataset(full_filenames,data_vars=vars_to_open)
    #    else:
    
    ds = xr.open_mfdataset(full_filenames) #,data_vars=vars)
    print(ds)

    ds['tmask'] = (('deptht','y','x'),tmask)
    if classification == 'physics':
        ds_phys = xr.open_mfdataset(input_path+'amm7_1d_*_grid_T.nc').rename_dims({'y_grid_T':'y','x_grid_T':'x'}).rename({'nav_lat_grid_T':'nav_lat','nav_lon_grid_T':'nav_lon'})
        ds = ds_phys[['votemper','vosaline']].isel(deptht=0)
        ds = xr.merge([ds, ds_phys['mldr10_1']])
    
    if classification == 'ecosys':
        ds['Phytoplankton'] = ds[['P1_c', 'P2_c', 'P3_c', 'P4_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['Zooplankton'] = ds[['Z4_c', 'Z5_c', 'Z6_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['DOC'] = ds[['R1_c', 'R2_c', 'R3_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['POC'] = ds[['R4_c', 'R6_c', 'R8_c']].to_array(dim='sum').sum(dim='sum', skipna=False)


    ds = ds[vars]
    print(ds)
    if depth == 'surface':
        ds=ds.isel(deptht=0)
        ds = ds.mean('time_counter')
    elif depth == 'bottom':
        # Apply mask and integrate
        # ds = (ds*grd.tmask).sum('deptht') DALE
        ds = (ds*grd.floor).sum('deptht')
    elif depth == 'DI':
        # multiply by thickness
        ds = ds*grd.e3t_0
        # Apply mask and integrate
        ds = (ds*grd.tmask).sum('deptht')

    ds = ds.isel(x=xsl,y=ysl)
    
    if depth != None:
        depth_name = depth + '_'
    else:
        depth_name = ''

    ds.to_netcdf('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_'+str(year_beg)+'-'+str(year_end)+'_'+depth_name+classification+'.nc')


