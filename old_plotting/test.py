import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import imageio.v2 as imageio

# 1. Find and sort all NetCDF files
file_pattern = "ICON-ART-STR_2019*.nc"
nc_files = sorted(glob.glob(file_pattern))
print(f" Found {len(nc_files)} files.")

# 2. Load them into one dataset along time
print("Opening dataset...")
ds = xr.open_mfdataset(nc_files, combine='by_coords')

# 3. Show available variables
print("\n Variables in dataset:")
for var in ds.data_vars:
    print(f" - {var}: {ds[var].dims}")

# 4. Ask user to pick one
# var_name = input("\n Enter variable name to plot: ").strip()
var_name = 'pntsrc'
var = ds[var_name]

# 5. Create folder for output
output_dir = f"frames_{var_name}"
os.makedirs(output_dir, exist_ok=True)

# 6. Save one plot per timestep
print("\n Generating plots...")
filenames = []
# Get consistent color scale
var2d_all = var.isel(height=0)
vmin = float(var2d_all.min())
vmax = float(var2d_all.max())
print(f"Using fixed color scale: min={vmin:.2f}, max={vmax:.2f}")
#print("pntsrc min:", float(var.min()))
#print("pntsrc max:", float(var.max()))
# Find all non-zero points
#nonzero = np.abs(var.values) > 1e-12
#print("Number of non-zero grid points:", nonzero.sum())
for i in range(var.sizes["time"]):
    # Extract full 3D field for this timestep
    var3d = var.isel(time=i)

    # Find the first height index (lowest) with any nonzero data
    # (This assumes 'height' is the second dimension: [height, lat, lon])
    height_index = None
    for h in range(var3d.sizes["height"]):
        if (var3d.isel(height=h).values != 0).any():
            height_index = h
            break

    if height_index is None:
        print(f"Skipping timestep {i}: all-zero data")
        continue

    # Now extract the 2D slice at that height
    data2d = var3d.isel(height=height_index)
    timestamp = str(data2d["time"].values)

    fig, ax = plt.subplots()
    data2d.plot(ax=ax, vmin=vmin, vmax=vmax, cmap="coolwarm")
    ax.set_title(f"{var_name} at {timestamp} (height={height_index})")
    ax.set_aspect('equal', adjustable='box')

    fname = f"{output_dir}/{var_name}_{i:04d}.png"
    plt.savefig(fname)
    plt.close(fig)
    filenames.append(fname)
#for i in range(var.sizes["time"]):
#    data2d = var.isel(time=i, height=0)
#    timestamp = str(data2d["time"].values)
#
#    fig, ax = plt.subplots()
#    data2d.plot(ax=ax, vmin=vmin, vmax=vmax, cmap="coolwarm")
#    ax.set_title(f"{var_name} at {timestamp}")
#    ax.set_aspect('equal', adjustable='box')
#
#    fname = f"{output_dir}/{var_name}_{i:04d}.png"
#    plt.savefig(fname)
#    plt.close(fig)
#    filenames.append(fname)

print(f"Saved {len(filenames)} frames in {output_dir}/")

# 7. Create MP4 animation
print("Creating animation...")
with imageio.get_writer(f"{var_name}.mp4", fps=4) as writer:
    for fname in filenames:
        image = imageio.imread(fname)
        writer.append_data(image)

print(f"Animation saved as {var_name}.mp4")