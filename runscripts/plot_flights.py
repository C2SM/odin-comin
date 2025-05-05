import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Optional: non-GUI backend for headless runs

import matplotlib.pyplot as plt
plt.clf()
plt.close('all')
from datetime import datetime

# Constants for conversion (molar mass ratio)
Mda, MCH4 = 28.964, 16.04

# Helper function to load and preprocess real flight data
def load_real_data(file_path):
    df = pd.read_csv(file_path, sep=';', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time_EPOCH'], unit='s')
    df['CH4_ppm'] = pd.to_numeric(df['CH4_ppm'], errors='coerce')
    return df[['Time', 'CH4_ppm']].dropna()

# Helper function to load and convert modeled CH4 data from mmr to vmr (in ppb)
def load_modeled_data(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['timepoint'])
    df['CH4'] = pd.to_numeric(df['CH4'], errors='coerce')
    # Convert from mass mixing ratio to volume mixing ratio in ppb
    df['CH4_ppb'] = df['CH4'] * (Mda / MCH4)
    # Densify to 1-second intervals and interpolate linearly
    df = df[['Time', 'CH4_ppb']].dropna()
    df = df.sort_values(by='Time').reset_index(drop=True)
    # df = df.set_index('Time').resample('50ms').interpolate().reset_index()

    return df

# Load all data
real7 = load_real_data("flight7.csv")
modeled7 = load_modeled_data("flight_modeled7.csv")

real8 = load_real_data("flight8.csv")
modeled8 = load_modeled_data("flight_modeled8.csv")


plt.close('all')  # <-- This clears any existing plots before creating new ones

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# Flight 7 Plot
axs[0].plot(real7['Time'], real7['CH4_ppm'] * 1000, label='Measured CH4 (ppb)', linewidth=1.5)
axs[0].plot(modeled7['Time'], modeled7['CH4_ppb'], label='Modeled CH4 (ppb)', linewidth=1.5, linestyle='-', marker='', solid_joinstyle='miter')
axs[0].set_title("Flight 7 - CH4 Comparison")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("CH4 (ppb)")
axs[0].legend()
axs[0].grid(True)

# Flight 8 Plot
axs[1].plot(real8['Time'], real8['CH4_ppm'] * 1000, label='Measured CH4 (ppb)', linewidth=1.5)
axs[1].plot(modeled8['Time'], modeled8['CH4_ppb'], label='Modeled CH4 (ppb)', linewidth=1.5, linestyle='-', marker='', solid_joinstyle='miter')
axs[1].set_title("Flight 8 - CH4 Comparison")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("CH4 (ppb)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("ch4_comparison.png", dpi=300)
plt.close(fig)  # Close the specific figure after saving
