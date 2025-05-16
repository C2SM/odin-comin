import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# File paths
satellite_nc_path = "TROPOMI_SRON_prs_flipped_20190101_20191231.nc"
model_csv_path = "satellite_data.csv"
output_figure_path = "comparison_heatmap_with_map.png"

# Load model data
model_df = pd.read_csv(model_csv_path)
model_df['timepoint'] = pd.to_datetime(model_df['timepoint'])
earliest_model_date = model_df['timepoint'].min()
latest_model_date = model_df['timepoint'].max()

# Load satellite data
nc = Dataset(satellite_nc_path, 'r')
obs = nc.variables['obs'][:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
date_raw = nc.variables['date'][:]
date_units = nc.variables['date'].units
sat_dates = num2date(date_raw, units=date_units)
valid_time_mask = np.array([((d <= latest_model_date) and (d >= earliest_model_date)) for d in sat_dates])
obs_filtered = obs[valid_time_mask]
lat_filtered = lat[valid_time_mask]
lon_filtered = lon[valid_time_mask]

# Create a 2D binned grid over lon/lat
def create_heatmap_data(values, lats, lons, bins=100):
    lon_bins = np.linspace(min(lons), max(lons), bins + 1)
    lat_bins = np.linspace(min(lats), max(lats), bins + 1)

    heatmap, _, _ = np.histogram2d(
        lats, lons, bins=[lat_bins, lon_bins], weights=values
    )
    counts, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
    with np.errstate(invalid='ignore', divide='ignore'):
        averaged = np.divide(heatmap, counts, out=np.full_like(heatmap, np.nan), where=counts != 0)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    return averaged.T, lon_centers, lat_centers

# Create heatmaps
sat_heatmap, sat_lons, sat_lats = create_heatmap_data(obs_filtered, lat_filtered, lon_filtered)
model_heatmap, model_lons, model_lats = create_heatmap_data(model_df['CH4'], model_df['lat'], model_df['lon'])

# Compute shared color scale
vmin = np.nanmin([np.nanmin(sat_heatmap), np.nanmin(model_heatmap)])
vmax = np.nanmax([np.nanmax(sat_heatmap), np.nanmax(model_heatmap)])

# Plotting with Cartopy
def plot_heatmap(ax, data, lons, lats, title):
    ax.set_title(title)
    ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                         shading='auto', vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.1)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)
    ax.gridlines(draw_labels=True)
    return mesh

fig, axs = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': ccrs.PlateCarree()})

mesh1 = plot_heatmap(axs[0], sat_heatmap, sat_lons, sat_lats, "Satellite Observations")
mesh2 = plot_heatmap(axs[1], model_heatmap, model_lons, model_lats, "Model Data")

# Colorbars with smaller width
cbar1 = fig.colorbar(mesh1, ax=axs[0], orientation='vertical', pad=0.01, shrink=0.8, fraction=0.03)
cbar1.set_label("Observed Value")

cbar2 = fig.colorbar(mesh2, ax=axs[1], orientation='vertical', pad=0.01, shrink=0.8, fraction=0.03)
cbar2.set_label("Model Value")

plt.tight_layout()
plt.savefig(output_figure_path)