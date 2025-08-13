import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta

nc_file_path = "output.nc"

# Load CH4 from .nc
def load_flight_data(nc_file):
    ds = Dataset(nc_file, 'r')

    base_time = datetime(1970, 1, 1)
    raw_times = ds.variables['etime'][:]
    time_vals = [base_time + timedelta(days=float(t)) for t in raw_times]

    site_name_raw = ds.variables['site_name'][:]
    site_names = [''.join(byte.decode('utf-8') for byte in char_array).strip() for char_array in site_name_raw]
    df = pd.DataFrame({
        'Time': time_vals,
        'site_name': site_names,
        'Temp': ds.variables['Temp'][:]
    })

    df = df[df['site_name'].str.startswith('flight')]
    return df.dropna(subset=['Temp', 'Time'])

# Load data
df_all = load_flight_data(nc_file_path)
flight_names = sorted(df_all['site_name'].unique())

# Prepare plot
fig, axs = plt.subplots(len(flight_names), 1, figsize=(14, 5 * len(flight_names)), sharex=False)
if len(flight_names) == 1:
    axs = [axs]

for i, flight in enumerate(flight_names):
    df_flight = df_all[df_all['site_name'] == flight]
    axs[i].plot(df_flight['Time'], df_flight['Temp'], label=f'Modeled Temperature (K) - {flight}', linewidth=1.5)

    axs[i].set_title(f"Temperature - {flight}")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Temp (Kelvin)")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig("temperature_flights.png", dpi=300)
plt.close(fig)