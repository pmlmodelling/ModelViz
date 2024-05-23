#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import netCDF4
import matplotlib.dates as mdates
from scipy import stats
import matplotlib as mpl
import numpy as np
import pylab as plt
import datetime
import pandas as pd
import itertools
import glob
import tqdm
import warnings

if __name__=="__main__":

    warnings.filterwarnings("ignore")

    from dask.distributed import Client
    client = Client()
    print(client.dashboard_link)

    # Note - which year/years - which depth - surface, bottom, depth integrated
    year_beg = 2001
    year_end = 2004

    depth = 'bottom'
    classification = 'biogeo'
   
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
    
    #vars = ['B1_c','Zooplankton', 'DOM', 'POM', 'Phytoplankton','N4_n','N5_s','O2_o','O3_c','O3_TA','N1_p','N3_n','votemper','vosaline']
    #vars = ['Zooplankton', 'fish_c_tot'] # 'fish_pelagic_size_spectrum_slope','
    # Map of names
    names={'votemper':'SST','vosaline':'SSS','N4_n':'Ammonium','N5_s':'Silicon','O2_o':'Oxygen','B1_c':'Bacteria','O3_c':'DIC',\
            'O3_TA':'Alkalinity','Zooplankton':'Zooplankton','DOM':'DOM','POM':'POM','Phytoplankton':'Phytoplankton','N1_p':'Phosphorous','fish_c_tot':'Fish biomass',\
            'fish_pelagic_size_spectrum_slope':'fish_size_spectrum_slope'} #,'N3_n':'Nitrogen',

    print('Loading data')
    #Load Data, combine some variables
    xsl = slice(15,-15)
    ysl = slice(15,-15)
    
    input_path = '/data/thaumus2/scratch/hpo/COMFORT/baseline_archerfull/rundata/200[0-4]/**/'
    full_filenames = input_path+'amm7_1d_*_ptrc_T.nc'
    ds = xr.open_mfdataset(full_filenames)
    
    if classification == 'physics':
        ds_phys = xr.open_mfdataset(input_path+'amm7_1d_*_grid_T.nc').rename_dims({'y_grid_T':'y','x_grid_T':'x'}).rename({'nav_lat_grid_T':'nav_lat','nav_lon_grid_T':'nav_lon'})
        ds = ds_phys[['votemper','vosaline']].isel(deptht=0)
        ds = xr.merge([ds, ds_phys['mldr10_1']])
    elif depth == 'surface':
        ds=ds.isel(deptht=0)
    elif depth == 'bottom':
        ds=ds.isel(deptht=-2)
    
    ds = ds.isel(x=xsl,y=ysl)

    if classification == 'ecosys':
        ds['Phytoplankton'] = ds[['P1_c', 'P2_c', 'P3_c', 'P4_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['Zooplankton'] = ds[['Z4_c', 'Z5_c', 'Z6_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['DOC'] = ds[['R1_c', 'R2_c', 'R3_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
        ds['POC'] = ds[['R4_c', 'R6_c', 'R8_c']].to_array(dim='sum').sum(dim='sum', skipna=False)

    ds = ds[vars]
    
    def mon_mean(x):
        return x.groupby('time_counter.month').mean('time_counter')

    # Calculate the monthly average
    ds = ds.resample(time_counter='MS').mean()
    #ds = ds.groupby("time_counter.year").apply(mon_mean) #.groupby("time_counter.month").mean(dim="time_counter")
    #ds = ds.stack(time_counter=("year","month"))
    #ds = ds.expand_dims(dim={"time":np.asarray([time_val])})

    if depth != None:
        depth_name = depth + '_'
    else:
        depth_name = ''

    ds.to_netcdf('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_monthly_mean_'+str(year_beg)+'-'+str(year_end)+'_'+depth_name+classification+'.nc')

