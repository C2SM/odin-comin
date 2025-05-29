import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta

# Constants
Mda, MCH4 = 28.964, 16.04
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
        'CH4_ppb': ds.variables['CH4'][:] * (Mda / MCH4)
    })

    df = df[df['site_name'].str.startswith('flight')]
    return df.dropna(subset=['CH4_ppb', 'Time'])

# Load real flight CSV
def load_real_csv(flight_csv_path):
    df = pd.read_csv(flight_csv_path, sep=';', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time_EPOCH'], unit='s')
    df['CH4_ppm'] = pd.to_numeric(df['CH4_ppm'], errors='coerce')
    return df[['Time', 'CH4_ppm']].dropna()

# Load data
df_all = load_flight_data(nc_file_path)
flight_names = sorted(df_all['site_name'].unique())

# Prepare plot
fig, axs = plt.subplots(len(flight_names), 1, figsize=(14, 5 * len(flight_names)), sharex=False)
if len(flight_names) == 1:
    axs = [axs]

for i, flight in enumerate(flight_names):
    # df_flight = df_all[df_all['site_name'] == flight].sort_values(by='Time')
    df_flight = df_all[df_all['site_name'] == flight]
    axs[i].plot(df_flight['Time'], df_flight['CH4_ppb'], label=f'Modeled CH4 (ppb) - {flight}', linewidth=1.5)

    # Try to infer flight date from time (e.g. 20191007)
    date_guess = df_flight['Time'].min().strftime("%Y%m%d")
    try:
        real_df = load_real_csv(f"/capstor/scratch/cscs/zhug/Romania6km/input/flights/flight{date_guess}.csv")
        axs[i].plot(real_df['Time'], real_df['CH4_ppm'] * 1000, label=f'Measured CH4 (ppb)', linewidth=1.5)
    except Exception as e:
        print(f"Could not load flight CSV for {flight} ({date_guess}): {e}")

    axs[i].set_title(f"CH4 Concentration - {flight}")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("CH4 (ppb)")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig("ch4_comparison_with_realdata.png", dpi=300)
plt.close(fig)