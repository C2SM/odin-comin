import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Find and sort all NetCDF files
file_pattern = "ICON-ART-STR_2019*.nc"
nc_files = sorted(glob.glob(file_pattern))
print(f" Found {len(nc_files)} files.")

# 2. Load dataset
print("Opening dataset...")
ds = xr.open_mfdataset(nc_files, combine='by_coords')

# 3. Show available variables
print("\n Variables in dataset:")
for var in ds.data_vars:
    print(f" - {var}: {ds[var].dims}")

# 4. Select variable
var_name = input("\n Enter variable name to plot: ").strip()
# var_name = 'pntsrc'
var = ds[var_name]

# 5. Output folder
output_dir = f"frames_{var_name}"
os.makedirs(output_dir, exist_ok=True)

# 6. Plot loop
print("\n Generating plots...")
filenames = []

# Color scale setup
var2d_all = var.isel(height=0)
vmin = float(var2d_all.min())
vmax = float(var2d_all.max())
print(f"Using fixed color scale: min={vmin:.2e}, max={vmax:.2e}")
# print("Number of non-zero grid points:", np.sum(np.abs(var.values) > 1e-12))

if var_name == "pntsrc":
    cmap = plt.cm.Reds.copy()
    cmap.set_under("white")
    norm = mcolors.Normalize(vmin=1e-12, vmax=vmax)
else:
    cmap = "coolwarm"
    norm = None

extpar = xr.open_dataset("../simulation_outer_domain/extpar_file.nc")
topo = extpar["TOPO_CLIM"]
topo_lon = np.degrees(extpar["clon"].values)
topo_lat = np.degrees(extpar["clat"].values)
lat_min = float(var.lat.min())
lat_max = float(var.lat.max())
lon_min = float(var.lon.min())
lon_max = float(var.lon.max())

for i in range(var.sizes["time"]):
    var3d = var.isel(time=i)

    # Find first non-zero height level
    height_index = None
    for h in range(var3d.sizes["height"]):
        if (np.abs(var3d.isel(height=h).values) > 1e-12).any():
            height_index = h
            break

    if height_index is None:
        print(f"Skipping timestep {i}: all-zero data")
        continue

    data2d = var3d.isel(height=height_index)
    timestamp = str(ds["time"].isel(time=i).values)

    fig = plt.figure(figsize=(10.08, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    data2d.plot(ax=ax, cmap=cmap, norm=norm)
    ax.set_title(f"{var_name} at {timestamp} (height={height_index})")
    # ax.set_aspect('equal', adjustable='box')

    # Overlay topography contours (optional but pretty)
    ax.tricontour(topo_lon, topo_lat, topo.values,
                levels=[500, 1000, 2000],
                colors="black", linewidths=0.5,
                transform=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    fname = f"{output_dir}/{var_name}_{i:04d}.png"
    plt.savefig(fname)
    plt.close(fig)
    filenames.append(fname)

print(f"Saved {len(filenames)} frames in {output_dir}/")

# 7. Create MP4 animation
print("Creating animation...")
with imageio.get_writer(f"{var_name}.mp4", fps=4) as writer:
    for fname in filenames:
        image = imageio.imread(fname)
        writer.append_data(image)

print(f"Animation saved as {var_name}.mp4")