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
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import tslearn as ts
import tslearn.utils
import tslearn.clustering
import tslearn.preprocessing
import warnings

if __name__=="__main__":

    warnings.filterwarnings("ignore")

    from dask.distributed import Client
    client = Client()
    print(client.dashboard_link)

    # Variables to train on
    vars = ['Y2_c','Y3_c','Y4_c','H1_c','H2_c','Q1_c','Q6_c']
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
    
    for year in range(2001,2005):
        for month in range(1,13):
            input_path = '/data/thaumus2/scratch/hpo/COMFORT/baseline_archerfull/'+str(year)+'/*'+str(month)+'/'
            print(input_path)
            ds = xr.open_mfdataset(input_path+'amm7_1d_'+str(year)+'*_ptrc_T.nc')
            # ds_phys = xr.open_mfdataset(input_path+'amm7_1d_'+str(year)+'*_grid_T.nc').rename_dims({'y_grid_T':'y','x_grid_T':'x'}).rename({'nav_lat_grid_T':'nav_lat','nav_lon_grid_T':'nav_lon'})
            #ds = xr.merge([ds,ds_phys[['votemper','vosaline']]]).isel(deptht=0,x=xsl,y=ysl)
            ds = ds.isel(x=xsl,y=ysl)

            #ds['Phytoplankton'] = ds[['P1_c', 'P2_c', 'P3_c', 'P4_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
            #ds['Zooplankton'] = ds[['Z4_c', 'Z5_c', 'Z6_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
            #ds['DOM'] = ds[['R1_c', 'R2_c', 'R3_c']].to_array(dim='sum').sum(dim='sum', skipna=False)
            #ds['POM'] = ds[['R4_c', 'R6_c', 'R8_c']].to_array(dim='sum').sum(dim='sum', skipna=False)

            ds = ds[vars]
            
            time_val = ds.time_counter.values[0]
            print(time_val)
            #month_length = ds.time_counter.dt.days_in_month
            # Calculate the weights by grouping by 'time.season'.
            #weights = (
            #    month_length.groupby("time_counter.month") / month_length.groupby("time_counter.month").sum()
            #)

            # Test that the sum of the weights for each season is 1.0
            #np.testing.assert_allclose(weights.groupby("time_counter.month").sum().values, np.ones(12))

            # Calculate the weighted average
            #ds_weighted = (ds * weights).groupby("time_counter.month").sum(dim="time_counter")
            #ds = ds_weighted.transpose('month','y','x')
            ds = ds.mean('time_counter')
            ds = ds.expand_dims(dim={"time":np.asarray([time_val])})

            ds.to_netcdf('/data/proteus1/scratch/rmi/classifications/COMFORT_data/benthic/amm7_1m_'+str(year)+'_'+str(month)+'.nc')


