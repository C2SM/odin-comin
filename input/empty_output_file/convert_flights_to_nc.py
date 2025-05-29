import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import sys
import os

def to_days_since_epoch(epoch_sec):
    return epoch_sec / 86400.0

def insert_flights_into_netcdf(nc_path, csv_files):
    with Dataset(nc_path, mode='a') as nc:

        obs_dim = nc.dimensions['obs']
        current_size = nc.variables['stime'].shape[0]

        obs_offset = current_size
        new_total = obs_offset

        for i, file in enumerate(csv_files):
            df = pd.read_csv(file, sep=';')
            n_new = len(df)
            new_total += n_new

            # Compute fields
            stime_days = df['Time_EPOCH'].astype(float).apply(to_days_since_epoch).to_numpy()
            etime_days = stime_days.copy()
            latitude = df['Latitude'].to_numpy()
            longitude = df['Longitude'].to_numpy()
            elevation = np.zeros_like(latitude)
            sampling_height = df['AGL_m'].to_numpy()
            sampling_strategy = np.full_like(latitude, 3)

            site_label = f"flight_{i+1}"
            site_name = np.array([list(site_label.ljust(20))] * n_new)

            nc.variables['stime'][obs_offset:new_total] = stime_days
            nc.variables['etime'][obs_offset:new_total] = etime_days
            nc.variables['latitude'][obs_offset:new_total] = latitude
            nc.variables['longitude'][obs_offset:new_total] = longitude
            nc.variables['elevation'][obs_offset:new_total] = elevation
            nc.variables['sampling_height'][obs_offset:new_total] = sampling_height
            nc.variables['sampling_strategy'][obs_offset:new_total] = sampling_strategy

            nc.variables['site_name'][obs_offset:new_total, :] = np.array(site_name, dtype='S1')

            obs_offset = new_total

nc_file = 'input.nc'

csv_files = list(Path('/capstor/scratch/cscs/zhug/Romania6km/input/flights').rglob('*.csv'))

insert_flights_into_netcdf(nc_file, csv_files)