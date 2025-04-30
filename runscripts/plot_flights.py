import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensures clean output in headless/bash environments
import matplotlib.pyplot as plt
from datetime import datetime

# Constants for conversion (molar mass ratio)
Mda, MCH4 = 28.964, 16.04

# Threshold for CH4 jump to break the line (in ppb)
CH4_JUMP_THRESHOLD = 2

# Helper function to load and preprocess real flight data
def load_real_data(file_path):
    df = pd.read_csv(file_path, sep=';', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time_EPOCH'], unit='s')
    df['CH4_ppm'] = pd.to_numeric(df['CH4_ppm'], errors='coerce')
    return df[['Time', 'CH4_ppm']].dropna()

# Helper function to load modeled data and convert to volume mixing ratio (ppb)
def load_modeled_data(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['timepoint'])
    df['CH4'] = pd.to_numeric(df['CH4'], errors='coerce')
    df['CH4_ppb'] = df['CH4'] * (Mda / MCH4)
    return df[['Time', 'CH4_ppb']].dropna()

# Plot modeled CH4 with jump-based segmentation
def plot_modeled(ax, df, label=None):
    diffs = df['CH4_ppb'].diff().abs()
    segment_ids = (diffs > CH4_JUMP_THRESHOLD).cumsum()
    for i, segment in df.groupby(segment_ids):
        ax.plot(segment['Time'], segment['CH4_ppb'], color='orange', linewidth=1.5, label=label if i == 0 else None)

# Load data
real7 = load_real_data("flight7.csv")
modeled7 = load_modeled_data("flight_modeled7.csv")

real8 = load_real_data("flight8.csv")
modeled8 = load_modeled_data("flight_modeled8.csv")

# Close previous figures
plt.clf()
plt.close('all')

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# Flight 7
axs[0].plot(real7['Time'], real7['CH4_ppm'] * 1000, label='Measured CH4 (ppb)', linewidth=1.5)
plot_modeled(axs[0], modeled7, label='Modeled CH4 (ppb)')
axs[0].set_title("Flight 7 - CH4 Comparison")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("CH4 (ppb)")
axs[0].legend()
axs[0].grid(True)

# Flight 8
axs[1].plot(real8['Time'], real8['CH4_ppm'] * 1000, label='Measured CH4 (ppb)', linewidth=1.5)
plot_modeled(axs[1], modeled8, label='Modeled CH4 (ppb)')
axs[1].set_title("Flight 8 - CH4 Comparison")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("CH4 (ppb)")
axs[1].legend()
axs[1].grid(True)

# Save
plt.tight_layout()
plt.savefig("ch4_comparison.png", dpi=300)
plt.close(fig)