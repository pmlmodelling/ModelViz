#!/usr/bin/env python
# coding: utf-8
# Preprocessing physics data - run preprocess_all_depths.py first.
import xarray as xr
import numpy as np
import warnings
import tqdm
import preprocess_amm7_functions as prep

if __name__=="__main__":

    
    # Note - which year/years - which depth - surface, bottom, depth integrated

    depth = 'DA'
    
    # foxed for this code
    classification = 'physics'
    year_beg = 2000
    year_end = 2004

    # Variables to train on
    vars = prep.cluster_vars(classification)

    print('Loading data')
    #Load Data, combine some variables
    xsl = slice(15,-15)
    ysl = slice(15,-15)

    input_path = '/data/proteus1/scratch/rmi/classifications/COMFORT_data/'
    full_filenames = input_path+'amm7_mean_2000-2004_all_depths_physics.nc'

    grd = xr.open_dataset('/data/sthenno1/to_archive/yuti/yuti-SSB-AMM7-hindcasts/mesh_mask.nc').isel(t=0)

    grd = prep.create_mask(grd,depth)

    for var_name in vars:
        # need to do each variable individually as otherwise uses too much memory
        ds = xr.open_mfdataset(full_filenames,chunks={'deptht':51,'x':100,'y':100},data_vars=[var_name])
        print(ds,flush=True)

       
        if var_name != 'mldr_10':
            if depth == 'surface':
                ds=ds.isel(deptht=0)
            elif depth == 'bottom':
                # Apply mask and integrate
                ds = (ds*grd.floor).sum('deptht')
            elif depth == 'DI':
                # multiply by thickness
                ds = ds*grd.e3t_0
                # Apply mask and integrate
                ds = (ds*grd.tmask).sum('deptht')
            elif depth == 'DA':
                # multiply by thickness
                ds = ds*grd.e3t_0
                # Apply mask and integrate
                ds = (ds*grd.tmask).sum('deptht')
                # Divide by deptht
                ds = ds/((grd.e3t_0*grd.tmask).sum('deptht'))
            elif type(depth) == int:
                # use nctoolkit to interpolate
                nc_ds = nc.from_xarray(ds)
                nc_grd = nc.from_xarray(grd.e3t_0)
                #print(nc_grd.e3t_0, flush=True)
                nc_ds.as_missing(0)
                nc_ds.vertical_interp(levels = [depth], thickness=nc_grd)
                nc_ds.missing_as(0)
                ds = nc_ds.to_xarray()
                ds = ds.isel(x=xsl,y=ysl)
                ds_bottom =xr.open_dataset('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_bottom_'+classification+'.nc')
                ds = xr.where(ds > 0, ds, ds_bottom[var_name])

        #ds = prep.select_depth(ds,grd,classification,depth)
        

        ds = ds[var_name]
        
        if type(depth) != int:
            ds = ds.isel(x=xsl,y=ysl)

            ds.name = var_name
        
        # for choosing a depth only: moved into function file
        #ds_bottom =xr.open_dataset('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_2000-2004_bottom_'+classification+'.nc')
        #print(ds_bottom, flush=True)

        #ds = xr.where(ds > 0, ds, ds_bottom[var_name])
        
        print(ds, flush=True)

        if var_name == vars[0]:
            ds_full = ds.copy()
        else:
            ds_full = xr.merge([ds_full,ds])

    if depth != None:
        depth_name = str(depth) + '_'
    else:
        depth_name = ''
   
    # add mixed layer depth after processing other variables
    ds = xr.open_mfdataset(full_filenames,chunks={'deptht':51,'x':100,'y':100},data_vars='mldr10_1')
    ds = ds.isel(x=xsl,y=ysl)
    #ds.name = var_name
    ds_full = xr.merge([ds_full,ds['mldr10_1']])

    ds_full.to_netcdf('/data/proteus1/scratch/rmi/classifications/COMFORT_data/amm7_mean_'+str(year_beg)+'-'+str(year_end)+'_'+depth_name+classification+'.nc')


