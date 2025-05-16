import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless runs

import matplotlib.pyplot as plt
from datetime import datetime

plt.clf()
plt.close('all')

# Constants for conversion (molar mass ratio)
Mda, MCH4 = 28.964, 16.04

# Helper function to load and preprocess real flight data
def load_real_data(file_path):
    df = pd.read_csv(file_path, sep=';', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time_EPOCH'], unit='s')
    df['CH4_ppm'] = pd.to_numeric(df['CH4_ppm'], errors='coerce')
    return df[['Time', 'CH4_ppm']].dropna()

# Helper function to load and convert modeled CH4 data
def load_modeled_data(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['timepoint'])
    df['CH4'] = pd.to_numeric(df['CH4'], errors='coerce')
    df['CH4_ppb'] = df['CH4'] * (Mda / MCH4)
    df = df[['Time', 'CH4_ppb']].dropna()
    return df

# Dates of the flights
flight_dates = ["20191007", "20191008"]

# Prepare figure
fig, axs = plt.subplots(len(flight_dates), 1, figsize=(14, 5 * len(flight_dates)), sharex=False)

for i, date_str in enumerate(flight_dates):
    real_file = f"flight{date_str}.csv"
    modeled_file = f"flight_modeled{date_str}.csv"

    real_df = load_real_data(real_file)
    modeled_df = load_modeled_data(modeled_file)

    axs[i].plot(real_df['Time'], real_df['CH4_ppm'] * 1000, label='Measured CH4 (ppb)', linewidth=1.5)
    axs[i].plot(modeled_df['Time'], modeled_df['CH4_ppb'], label='Modeled CH4 (ppb)', linewidth=1.5)
    axs[i].set_title(f"Flight {date_str} - CH4 Comparison")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("CH4 (ppb)")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig("ch4_comparison_flights.png", dpi=300)
plt.close(fig)