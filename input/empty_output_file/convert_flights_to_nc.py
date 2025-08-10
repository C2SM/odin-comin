import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path

def to_days_since_epoch(epoch_sec):
    return epoch_sec / 86400.0

def create_netcdf_with_data(nc_path, csv_files):
    # First gather all data
    all_data = []
    total_obs = 0

    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, sep=';')
        n = len(df)
        total_obs += n

        stime_days = df['Time_EPOCH'].astype(float).apply(to_days_since_epoch).to_numpy()
        etime_days = stime_days.copy()
        latitude = df['Latitude'].to_numpy()
        longitude = df['Longitude'].to_numpy()
        elevation = np.zeros_like(latitude)
        sampling_height = df['AGL_m'].to_numpy()
        sampling_strategy = np.full(n, 3, dtype=np.float32)
        site_label = f"flight_{i+1}"
        site_name = np.array([list(site_label.ljust(20))] * n)

        all_data.append({
            'stime': stime_days,
            'etime': etime_days,
            'latitude': latitude,
            'longitude': longitude,
            'elevation': elevation,
            'sampling_height': sampling_height,
            'sampling_strategy': sampling_strategy,
            'site_name': site_name,
            'n': n
        })

    # Create the NetCDF file with unlimited obs dimension
    with Dataset(nc_path, mode='w') as nc:
        nc.createDimension('obs', None)       # unlimited
        nc.createDimension('nchar', 20)

        # Variables
        stime = nc.createVariable('stime', 'f8', ('obs',))
        stime.units = "days since 1970-01-01 00:00:00"
        stime.long_name = "start time of observation interval; UTC"
        stime.calendar = "proleptic_gregorian"

        etime = nc.createVariable('etime', 'f8', ('obs',))
        etime.units = stime.units
        etime.long_name = "end time of observation interval; UTC"
        etime.calendar = "proleptic_gregorian"

        lon = nc.createVariable('longitude', 'f4', ('obs',), fill_value=1.0e+20)
        lon.units = "degrees_east"
        lon.standard_name = "longitude"

        lat = nc.createVariable('latitude', 'f4', ('obs',), fill_value=1.0e+20)
        lat.units = "degrees_north"
        lat.standard_name = "latitude"

        elev = nc.createVariable('elevation', 'f4', ('obs',), fill_value=1.0e+20)
        elev.units = "m"
        elev.long_name = "surface elevation above sea level"

        sh = nc.createVariable('sampling_height', 'f4', ('obs',), fill_value=1.0e+20)
        sh.units = "m"
        sh.long_name = "sampling height above surface"

        ss = nc.createVariable('sampling_strategy', 'f4', ('obs',))
        ss.units = "1"
        ss.long_name = "sampling strategy flag"
        ss.comment = "1=low ; 2=mountain ; 3=flight"

        sn = nc.createVariable('site_name', 'S1', ('obs', 'nchar'))
        sn.long_name = "station name or ID"

        # Global attributes
        nc.Conventions = "CF-1.8"
        nc.title = "Station input file for ICON ComIn interface XYZ"
        nc.institution = "Empa"
        nc.source = "ICON ComIn interface XYZ"
        nc.version = "1.0"
        nc.author = "Zeno Hug"
        nc.transport_model = "ICON"
        nc.transport_model_version = ""
        nc.experiment = ""
        nc.project = ""
        nc.references = ""
        nc.comment = ""
        nc.license = "CC-BY-4.0"
        nc.history = ""

        # Write all data
        offset = 0
        for d in all_data:
            n = d['n']
            slc = slice(offset, offset + n)
            nc.variables['stime'][slc] = d['stime']
            nc.variables['etime'][slc] = d['etime']
            nc.variables['latitude'][slc] = d['latitude']
            nc.variables['longitude'][slc] = d['longitude']
            nc.variables['elevation'][slc] = d['elevation']
            nc.variables['sampling_height'][slc] = d['sampling_height']
            nc.variables['sampling_strategy'][slc] = d['sampling_strategy']
            nc.variables['site_name'][slc, :] = np.array(d['site_name'], dtype='S1')
            offset += n

# 🔧 Usage
csv_files = list(Path('/capstor/scratch/cscs/zhug/Romania6km/input/flights').rglob('*.csv'))
nc_file = 'input_flight.nc'
create_netcdf_with_data(nc_file, csv_files)