import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Path to your NetCDF file
filename = 'tracked_ch4.nc'

# Open NetCDF file
ds = nc.Dataset(filename)

# Read time and CH4 values
time_var = ds.variables['time'][:]
ch4_var = ds.variables['avg_CH4'][0, :]  # Assuming one station

# Convert time from seconds since 2019-01-01 to datetime
start_time = datetime(2019, 1, 1)
time = [start_time + timedelta(seconds=int(t)) for t in time_var]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time, ch4_var, marker='o', linestyle='-', markersize=3)  # Reduced markersize
plt.xlabel('Time')
plt.ylabel('CH4 concentration (ppb)')
plt.title('CH4 Concentration Over Time')
plt.grid(True)
plt.tight_layout()

# Save plot
output_file = 'ch4_station_example.png'
plt.savefig(output_file)

print(f"Plot saved to {output_file}")