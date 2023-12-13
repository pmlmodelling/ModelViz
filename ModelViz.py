#!/bin/python

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import numpy as np
import pandas as pd
import pathlib
import tslearn as ts
import tslearn.clustering
import glob
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec

class ModelViz:
    def __init__(self):
        self.cluster_vars = ['N1_p', 'N3_n', 'N4_n', 'N5_s', 'O2_o', 'B1_c','O3_c', 'O3_TA', 'O3_pH', 'Phytoplankton', 'Zooplankton', 'DOM', 'POM']
        self.time_var = 'time_counter'
        self.x_strip = slice(10,-10)
        self.y_strip = slice(10,-10)
        self.n_clusters = 6
        self.norm = 'magnitude' # other options None and stdev
        self.seed = 950
        self.n_init = 3
    
    def load_data(self, file_path):
        self.ds = xr.open_dataset(file_path).drop_dims('axis_nbounds', errors='ignore')
        if 'deptht' in self.ds.dims:
            self.ds = self.ds.squeeze(dim=['deptht'])
        
    def load_mfdata(self, file_glob):
        files = glob.glob(file_glob)
        self.ds = xr.open_mfdataset(files).drop_dims('axis_nbounds', errors='ignore')
        if 'deptht' in self.ds.dims:
            self.ds = self.ds.squeeze(dim=['deptht'])

    def load_grid(self, file_path):
        self.grd = xr.open_dataset(file_path).squeeze(dim=['t']).isel(x=self.x_strip,y=self.y_strip)
        self.dim_x = self.grd.x
        self.dim_y = self.grd.y
        self.mask = xr.where(self.grd.bottom_level>0,1,0) #.stack(Npts=('x','y'))
        
    def save_data(self, file_path):
        self.ds.to_netcdf(file_path)        
    
    def summarise_features(self, sum_vars=None):
        if sum_vars is None:
            sum_vars={
                'Phytoplankton': ['P1_c', 'P2_c', 'P3_c', 'P4_c'],
                'Zooplankton': ['Z4_c', 'Z5_c', 'Z6_c'],
                'DOM': ['R1_c', 'R2_c', 'R3_c'],
                'POM': ['R4_c', 'R6_c', 'R8_c']
                }
        for v in sum_vars:
            self.ds[v] = self.ds[sum_vars[v]].to_array(dim='sum').sum(dim='sum', skipna=False)
        
    def preprocess(self, do_slice=True):
        self.ds = self.ds[self.cluster_vars]
        if do_slice == True:
            self.ds = self.ds.isel(x=self.x_strip, y=self.y_strip)
        if self.time_var in self.ds.dims:
            self.ds = self.ds.rename({self.time_var:'time'})
        else:
            self.ds = self.ds.expand_dims(dim = {"time":np.asarray([1])})
        self.ds = self.ds.where(self.mask==1,drop=True)
        if self.norm == 'magnitude':
            # Global normalisation by magnitude of variable for all data
            self.norm_factor = {}
            for v in self.cluster_vars:
                self.norm_factor[v] = np.sqrt((self.ds[v]*self.ds[v]).sum())
                self.ds[v] = self.ds[v]/self.norm_factor[v]
        if self.norm == 'stdev':
            # Global normalisation by magnitude and variability for point data
            for v in self.cluster_vars:
                self.ds[v] = (self.ds[v]-self.ds[v].mean())/self.ds[v].std()

    def make_tsds(self, save=False, file_path='dataset.csv'):
        # not sure why I currently need to mask again here?
        ds_stack = self.ds.stack(Npts=('x','y')).where(self.mask.stack(Npts=('x','y'))==1,drop=True)
        self.index = ds_stack.Npts
        ds_stack = ds_stack.to_stacked_array('z', sample_dims=['Npts'])
        self.tsds = pd.DataFrame(ds_stack.variable, index=self.index, columns=ds_stack.time)
        if save:
            self.tsds.to_csv(pathlib.Path(file_path))
    
    def load_tsds(self, file_path):
        self.tsds = pd.read_csv(file_path)
        self.index = self.tsds.index
    
    def train(self, tsds=None, n_clusters=6, method='quantile', verbose=True, save=True, file_path='model.ks', model_name='kshape'):
        '''Train model using either kshape (for time series data) or kmeans (for single time point data)'''
        if tsds is None:
            tsds = self.tsds
        self.n_clusters = n_clusters
        if model_name == 'kmeans':
            self.model = KMeans(init="k-means++", n_clusters=n_clusters, n_init=self.n_init, random_state=self.seed)
        if model_name == 'kshape':
            if method == 'quantile':
                print('Initialising using quantiles')
                quantiles = np.arange(1/(2*n_clusters),1,1/n_clusters)
                self.model = ts.clustering.KShape(n_clusters=n_clusters,
                                           verbose=verbose,
                                           init=tsds.quantile(q=quantiles).values[:,:,np.newaxis],
                                           n_init = 1
                                           )
            elif method == 'random':
                print('Initialising using random, seed = ',self.seed)
                self.model = ts.clustering.KShape(n_clusters=n_clusters,
                                   verbose=verbose,
                                   random_state=self.seed,
                                   n_init = self.n_init
                                   )
            else:
                print('Unrecognised initialisation method')
                return
        self.model.fit(tsds)
        if save:
            self.model.to_json(file_path)

    def load_model(self, file_path):
        self.model = ts.clustering.KShape.from_json(file_path)

    def predict(self):
        pred_labels = self.model.predict(self.tsds)
        predictions = pd.DataFrame(pred_labels, index=self.index, columns=['Clusters'])
        predictions.index = pd.MultiIndex.from_tuples(predictions.index,names=('x','y'))
        # Restore original dimensions:
        if len(predictions.index.get_level_values(0).unique()) != len(self.dim_x):
            for idx in np.setdiff1d(self.dim_x,predictions.index.get_level_values(0).unique()):
                predictions.loc[(idx,0),:] = np.nan
        if len(predictions.index.get_level_values(1).unique()) != len(self.dim_y):
            for idx in np.setdiff1d(self.dim_y,predictions.index.get_level_values(1).unique()):
                predictions.loc[(0,idx),:] = np.nan
        self.predictions = predictions.to_xarray()
    
    def get_cluster_info(self, save=False, file_path='Predicted_TS.nc'):
        w = self.predictions.Clusters
        self.cluster_ds = xr.Dataset(data_vars={'class_map':(['y','x'],w.values.T)},
                                coords={'lon':(['y','x'],self.grd.nav_lon.values),
                                    'lat':(['y','x'],self.grd.nav_lat.values)})

        self.cluster_ds=self.cluster_ds.assign_coords({'var':self.cluster_vars,'vclass':np.arange(self.model.n_clusters),'time':self.ds.time.values})
        self.cluster_ds['class_TS'] = (['var','vclass','time'], np.zeros((len(self.cluster_vars),self.model.n_clusters,len(self.ds.time))))
        self.cluster_ds['class_TS_std'] = (['var','vclass','time'], np.zeros((len(self.cluster_vars),self.model.n_clusters,len(self.ds.time))))
        self.cluster_ds['IQR'] = (['var'],np.zeros((len(self.cluster_vars))))
        self.ds = self.ds.chunk(dict(time=-1))
        for v,var in enumerate(self.cluster_vars):
            print('Processing '+var)
            self.cluster_ds['IQR'][v] = self.ds[var].quantile([0.75,0.25]).diff('quantile').values[0]
            for x in tqdm.tqdm(np.arange(self.n_clusters)):
                self.cluster_ds['class_TS'][v,x,:] = self.ds[var].where((w==x)).mean(['x','y'])
                self.cluster_ds['class_TS_std'][v,x,:] = self.ds[var].where((w==x)).std(['x','y'])
        if save:
            self.cluster_ds.to_netcdf(file_path)

    
    def load_cluster_info(self, file_path):
        self.cluster_ds = xr.open_dataset(file_path)

    def plot_map(self, savefig=None, file_path='class_map.png'):
        self.cmap = plt.get_cmap('Set3')

        f = plt.figure(figsize=(8,8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.pcolormesh(self.cluster_ds.lon,self.cluster_ds.lat,self.cluster_ds.class_map,cmap=self.cmap)
        ax.add_feature(NaturalEarthFeature(category="physical", facecolor=[0.9, 0.9, 0.9], name="coastline", scale="50m"), edgecolor="gray")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", linestyle="-")
        gl.top_labels = False
        gl.bottom_labels = True
        gl.right_labels = False
        gl.left_labels = True
        ax.set_aspect("auto")
        if savefig is not None:
            plt.savefig(file_path)
            
    def plot_ts(self, plot_vars={'N3_n':'Nitrate','Phytoplankton':'Phyto', 'DOM':'DOM', 'POM':'POM'}, rescale=False, savefig=None, file_path='cluster_ts.png'):
        self.line_colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        self.cmap = plt.get_cmap('Set3')
        self.cmap_discrete = self.cmap(np.linspace(0,1,self.model.n_clusters))
        
        for i in range(self.model.n_clusters):
            f = plt.figure(figsize=(8,3))
            ax = plt.axes()
            for v,var in enumerate(plot_vars):
                scale = self.norm_factor[var] if rescale else 1
                xbar = scale*self.cluster_ds.class_TS.sel(var=var,vclass=i)
                xstd = scale*self.cluster_ds.class_TS_std.sel(var=var,vclass=i)
                ax.plot(self.cluster_ds.time,xbar,label=plot_vars[var],c=self.line_colors[v])
                ax.fill_between(self.cluster_ds.time,xbar-xstd,xbar+xstd,alpha=0.1,facecolor=self.line_colors[v]) 
            ax.set_xlim([self.cluster_ds.time[0],self.cluster_ds.time[-1]])
            ax.tick_params(axis='x', labelrotation=45)
            ax.legend(loc='upper right',ncol=2)
            xmin,xmax = plt.gca().get_xlim()
            xdiff = xmax-xmin
            ymin,ymax = plt.gca().get_ylim()
            ydiff = ymax-ymin
            ax.add_patch(mpl.patches.Rectangle((xmin+0.01*xdiff,ymin+0.82*ydiff),0.05*xdiff,0.15*ydiff,facecolor=self.cmap_discrete[i]))    
            if savefig is not None:
                p = pathlib.Path(file_path)
                plt.savefig(p.with_stem(f"{p.stem}_{i}"))
    
    def plot_vars(self, plot_vars={'N3_n':'Nitrate','Phytoplankton':'Phyto', 'DOM':'DOM', 'POM':'POM'}, rescale=False, savefig=None, file_path='cluster_ts.png'):
        # for plotting cluster outputs when input variables are not time series
        self.line_colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        self.cmap = plt.get_cmap('Set3')
        self.cmap_discrete = self.cmap(np.linspace(0,1,self.model.n_clusters))
        gs = gridspec.GridSpec(self.model.n_clusters, 1)
        f = plt.figure(figsize=(8,3*self.model.n_clusters))
        for i in range(self.model.n_clusters):
            ax = f.add_subplot(gs[i])
            x_tick_labels = np.asarray([])
            for v,var in enumerate(plot_vars):
                scale = self.norm_factor[var] if rescale else 1
                xbar = scale*self.cluster_ds.class_TS.sel(var=var,vclass=i)
                xstd = scale*self.cluster_ds.class_TS_std.sel(var=var,vclass=i)
                ax.plot(v+1,xbar,'o',label=plot_vars[var],c=self.line_colors[v])
                ax.errorbar(v+1,xbar,xstd,alpha=0.5,color=self.line_colors[v]) 
                x_tick_labels = np.append(x_tick_labels,[plot_vars[var]])
            ax.set_xlim([0,v+2])
            ax.set_xticks(range(1,v+2),x_tick_labels)
            xmin,xmax = plt.gca().get_xlim()
            xdiff = xmax-xmin
            ymin,ymax = plt.gca().get_ylim()
            ydiff = ymax-ymin
            ax.add_patch(mpl.patches.Rectangle((xmin+0.01*xdiff,ymin+0.82*ydiff),0.05*xdiff,0.15*ydiff,facecolor=self.cmap_discrete[i]))    
        if savefig is not None:
            p = pathlib.Path(file_path)
            plt.savefig(p.with_stem(f"{p.stem}"))  
    
