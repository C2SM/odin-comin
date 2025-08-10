import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless environments

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np

# === CONFIGURATION ===
file_name = "output.nc"  # Replace with the actual path to your NetCDF file

# === LOAD DATA ===
with Dataset(file_name, mode='r') as nc:
    etime = nc.variables['etime'][:]
    time_units = nc.variables['etime'].units
    calendar = nc.variables['etime'].calendar

    # Convert numeric times to datetime objects
    # dates = num2date(etime, units=time_units, calendar=calendar)
    dates = num2date(etime, units=time_units, calendar=calendar)

    # Convert to datetime.datetime objects
    from datetime import datetime
    dates = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]

    # Load CH4 and Temperature data
    ch4 = nc.variables['CH4'][:]
    temp = nc.variables['Temp'][:]

# === PLOT CH4 ===
plt.figure()
plt.plot(dates, ch4, label='CH4 (ppb)')
plt.title('CH₄ Concentration Time Series')
plt.xlabel('Time')
plt.ylabel('CH₄ Concentration (ppb)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('ch4_timeseries.png')

# === PLOT TEMPERATURE ===
plt.figure()
plt.plot(dates, temp, label='Temperature (K)', color='orange')
plt.title('Temperature Time Series')
plt.xlabel('Time')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('temperature_timeseries.png')