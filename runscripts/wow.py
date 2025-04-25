import xarray as xr
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

ds = xr.open_dataset("tracked_ch4.nc")

# Loop over all stations
for i, station_name in enumerate(ds["station"].values):
    fig, ax = plt.subplots()
    ds["avg_CH4"].isel(station=i).plot(ax=ax)
    ax.set_title(f"CH4 Time Series - {station_name}")
    ax.set_ylabel("CH4 concentration (ppb)")
    ax.set_xlabel("Time")

    # Save the figure
    filename = f"ch4_timeseries_{station_name}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)  # Close to free memory in batch runs