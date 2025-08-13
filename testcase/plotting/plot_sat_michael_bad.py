import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from netCDF4 import Dataset, num2date
import matplotlib as mpl

# === FILE PATHS ===
satellite_nc_path = "/capstor/scratch/cscs/zhug/Romania6km/input/TROPOMI/TROPOMI_SRON_corners_20190101_20191231.nc"
model_nc_path = "output_sat_ch4.nc"
output_figure_path = "comparison_plot.png"


# === LOAD DATA ===
f_sat = Dataset(satellite_nc_path)
f_mod = Dataset(model_nc_path)

# Read time and convert to datetime
sat_dates = num2date(f_sat.variables["date"][:], f_sat.variables["date"].units)
mod_dates = num2date(f_mod.variables["date"][:], f_mod.variables["date"].units)
latest_model_date = max(mod_dates)

# Create time mask
time_mask = np.array(sat_dates) <= latest_model_date

# Apply mask to satellite data
obs_sat = f_sat.variables["obs"][:][time_mask]
lat_corners = f_sat.variables["latitude_bounds"][:, time_mask]
lon_corners = f_sat.variables["longitude_bounds"][:, time_mask]

# Model data (same order, assumed)
obs_mod = f_mod.variables["CH4"][:]

# === Ensure matching shapes ===
assert obs_sat.shape[0] == obs_mod.shape[0], "Mismatch in filtered satellite and model data lengths."

# Create polygons
n_obs = obs_sat.shape[0]
polygons = [Polygon(zip(lon_corners[:, i], lat_corners[:, i])) for i in range(n_obs)]

# Compute difference
diff = obs_mod - obs_sat
maxdiff = np.nanquantile(np.abs(diff), 0.994)

# === CREATE GEODATAFRAMES ===
gdf_sat = gpd.GeoDataFrame({'geometry': polygons, 'xch4': obs_sat})
gdf_mod = gpd.GeoDataFrame({'geometry': polygons, 'xch4': obs_mod})
gdf_diff = gpd.GeoDataFrame({'geometry': polygons, 'xch4': diff})

for gdf in [gdf_sat, gdf_mod, gdf_diff]:
    gdf.set_crs(epsg=4326, inplace=True)

# === PLOTTING ===
vmin, vmax = 1810, 1860
cmap = plt.cm.viridis
cmapd = plt.cm.bwr

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax1, ax2, ax3 = axes

def plot_panel(ax, gdf, title, cmap, vmin, vmax):
    ax.set_title(title, fontsize=15)
    gdf.plot(column='xch4', cmap=cmap, vmin=vmin, vmax=vmax,
             linewidth=0, edgecolor='none', ax=ax)
    ax.set_xlim(19.3, 30.3)
    ax.set_ylim(43.05, 48.75)
    ax.set_aspect('equal')

# Plot each panel
plot_panel(ax1, gdf_sat, "TROPOMI", cmap, vmin, vmax)
plot_panel(ax2, gdf_mod, "ICON-ART", cmap, vmin, vmax)
plot_panel(ax3, gdf_diff, "ICON - TROPOMI", cmapd, -maxdiff, maxdiff)

# Add colorbars
for ax, cmap_, vmin_, vmax_, label in zip(
    [ax1, ax2, ax3],
    [cmap, cmap, cmapd],
    [vmin, vmin, -maxdiff],
    [vmax, vmax, maxdiff],
    ["XCH4 (ppb)", "XCH4 (ppb)", "ΔXCH4 (ppb)"]
):
    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=mpl.colors.Normalize(vmin=vmin_, vmax=vmax_))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.01)
    cbar.set_label(label, size='large')

plt.tight_layout()
plt.savefig(output_figure_path, dpi=300, bbox_inches='tight')
plt.close()