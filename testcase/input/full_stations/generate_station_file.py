from netCDF4 import Dataset, date2num
import numpy as np
from datetime import datetime, timedelta

# Time setup
start_time = datetime(2019, 1, 1, 0, 0)
end_time = datetime(2019, 1, 10, 0, 0)
time_unit = "days since 1970-01-01 00:00:00"
calendar = "proleptic_gregorian"

# Output file
output_file = "station_input.nc"

# Time list
num_hours = int((end_time - start_time).total_seconds() // 3600)
times = [start_time + timedelta(hours=i) for i in range(num_hours)]

# Site values
lon_val = 24.0
lat_val = 45.0
elevation_val = 100.0
sampling_height_val = 2.0
sampling_strategy_val = 1
site_name_str = "TEST"
nchar = 20

# Create NetCDF
with Dataset(output_file, "w", format="NETCDF4") as nc:

    # Dimensions
    nc.createDimension("obs", None)
    nc.createDimension("nchar", nchar)

    # Variables
    stime = nc.createVariable("stime", "f8", ("obs",))
    stime.units = time_unit
    stime.long_name = "start time of observation interval; UTC"
    stime.calendar = calendar

    etime = nc.createVariable("etime", "f8", ("obs",))
    etime.units = time_unit
    etime.long_name = "end time of observation interval; UTC"
    etime.calendar = calendar

    lon = nc.createVariable("longitude", "f4", ("obs",))
    lon.units = "degrees_east"
    lon.standard_name = "longitude"

    lat = nc.createVariable("latitude", "f4", ("obs",))
    lat.units = "degrees_north"
    lat.standard_name = "latitude"

    elev = nc.createVariable("elevation", "f4", ("obs",))
    elev.units = "m"
    elev.long_name = "surface elevation above sea level"

    samp_h = nc.createVariable("sampling_height", "f4", ("obs",))
    samp_h.units = "m"
    samp_h.long_name = "sampling height above surface"

    samp_strat = nc.createVariable("sampling_strategy", "f4", ("obs",))
    samp_strat.units = "1"
    samp_strat.long_name = "sampling strategy flag"
    samp_strat.comment = "1=low ; 2=mountain ; 3=instantaneous low; 4=instantaneous mountain"

    site_name = nc.createVariable("site_name", "S1", ("obs", "nchar"))
    site_name.long_name = "station name or ID"

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

    # Data
    stime[:] = date2num(times, time_unit, calendar)
    etime[:] = date2num([t + timedelta(hours=1) for t in times], time_unit, calendar)
    lon[:] = lon_val
    lat[:] = lat_val
    elev[:] = elevation_val
    samp_h[:] = sampling_height_val
    samp_strat[:] = sampling_strategy_val

    # Repeat site_name for each obs
    site_bytes = np.array([c.encode('utf-8') for c in site_name_str.ljust(nchar)])
    site_name[:, :] = np.tile(site_bytes, (num_hours, 1))