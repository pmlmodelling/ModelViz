import nctoolkit as nc
import numpy as np

def cluster_vars(classification):
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
        vars = ['votemper','vosaline','PAR']
    #vars = ['Zooplankton', 'fish_c_tot'] # 'fish_pelagic_size_spectrum_slope','
    return vars

def create_mask(grd,depth):
    if depth == 'DI':
        # Create 3D mask for NEMO 4.0 - not needed for NEMO 3.6 as this is already provided
        if not 'tmask' in grd.variables:
            tmask = 0*grd.e3t_0.values + 1
            for i in tqdm.tqdm(grd.x):
                for j in grd.y:
                    tmask[int(grd.bottom_level.isel(x=i,y=j).values):,j,i] = 0
            grd['tmask'] = (('z','y','x'),tmask)
    
    if depth == 'bottom':
        #Create 3D mask
        # NEMO 4.0
        if 'bottom_level' in grd.variables:
            floor = 0*grd.e3t_0.values
            for i in tqdm.tqdm(grd.x):
                for j in grd.y:
                    floor[int(grd.bottom_level.isel(x=i,y=j).values)-1,j,i] = 1
            grd['floor'] = (('z','y','x'),floor)
        # NEMO 3.6
        if 'tmask' in grd.variables:
            floor = 0*grd.tmask.values
            for i in np.arange(0,floor.shape[1]):
              for j in np.arange(0,floor.shape[2]):
                for k in np.arange(-1,-51,-1):
                  if grd.tmask[k,i,j]==1: floor[k,i,j]=1; break
            grd['floor'] = (('z','y','x'),floor)

    # Rename depth in grd dataset
    grd = grd.rename_dims({'z':'deptht'})
    return grd

def select_depth(ds,grd,classification,depth):
    if depth == 'surface':
        if classification != 'physics':
            ds=ds.isel(deptht=0)
        ds = ds.mean('time_counter')
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
    return ds
