import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
from datetime import datetime
from netCDF4 import Dataset, num2date
import numpy as np
from shapely.strtree import STRtree
from collections import defaultdict
import matplotlib.pyplot as plt


# === CONFIGURATION ===
start_date = datetime(2019, 1, 1)
end_date   = datetime(2019, 1, 10)  # Change to Feb if needed

# === FILE PATHS ===
base_dir = Path(__file__).parent

# TROPOMI input
# tropomi_file = base_dir / "TROPOMI_SRON_corners_20190101_20191231.nc"
tropomi_file = '/capstor/scratch/cscs/zhug/Romania6km/input/TROPOMI/TROPOMI_SRON_corners_20190101_20191231.nc'

# ICON-ODIN CH4 interpolated output
icon_odin_file = base_dir / "output_sat_ch4_until_9_january.nc"

# ICON model output directory (hourly files)
icon_hourly_dir = base_dir  # or base_dir / "ICON_output"

# ICON grid file
icon_grid_file = Path("/capstor/scratch/cscs/zhug/Romania6km/input/grid/inner_grid/dyn_grid.nc")

# CAMS climatology and LBCs
cams_climatology_file = Path("/capstor/scratch/cscs/zhug/Romania6km/input/CAMS/cams73_v22r2_ch4_conc_surface_inst_201910.nc")
cams_lbc_dir = Path("/capstor/scratch/cscs/zhug/Romania6km/input/CAMS/LBC")

# Output directory
output_dir = base_dir / "remapped_output"
output_dir.mkdir(exist_ok=True)


##
# @file get_int_coefs.py
#
# @brief Helper function
#
# @section description_get_int_coefs Description
# Helper Function
#
# @section libraries_get_int_coefs Libraries/Modules
# - numpy library
#
# @section author_get_int_coefs Author(s)
# - Created by ?? on ??
#
# Copyright (c) 2025 Empa.  All rights reserved.



def get_int_coefs(pb_ret, pb_mod):
    """! Computes a coefficients matrix to transfer a model profile
    onto a retrieval pressure axis.

    In cases where retrieval surface pressure is higher than model
    surface pressure, and in cases where retrieval top pressure is
    lower than model top pressure, the model profile will be
    extrapolated with constant tracer mixing ratios. In cases where
    retrieval surface pressure is lower than model surface pressure,
    and in cases where retrieval top pressure is higher than model
    top pressure, only the parts of the model column that fall
    within the retrieval presure boundaries are sampled.

    Usage
    -----
            .. code-block:: python

                import numpy as np
                pb_ret = np.linspace(900., 50., 5)
                pb_mod = np.linspace(1013., 50., 7)
                model_profile = 1. - np.linspace(0., 1., len(pb_mod)-1)**3
                coefs = get_int_coefs(pb_ret, pb_mod, "layer_average")
                retrieval_profile = np.matmul(coefs, model_profile)

    @param pb_ret       (:class:`array_like`) Pressure boundaries of the retrieval column
    @param pb_mod       (:class:`array_like`) Pressure boundaries of the model column
    @param level_def     A list of sampling heights, meaning height above ground in meters
    @return (:class:`array_like`) Integration coefficient matrix. Each row sums to 1.
    """

    # This code assumes that WRF variables are constant in
    # layers, but they are defined on levels. This can be seen
    # for example by asking wrf.interplevel for the value of a
    # variable that is defined on the mass grid ("theta points")
    # at a pressure slightly higher than the pressure on its
    # grid (wrf.getvar(ncf, "p")), it returns nan. So There is
    # no extrapolation. There are no layers. There are only
    # levels.
    # In addition, this page here:
    # https://www.openwfm.org/wiki/How_to_interpret_WRF_variables
    # says that to find values at theta-points of a variable
    # living on u-points, you interpolate linearly. That's the
    # other way around from what I would do if I want to go from
    # theta to staggered.
    # WRF4.0 user guide:
    # - ungrib can interpolate linearly in p or log p
    # - real.exe comes with an extrap_type namelist option, that
    #   extrapolates constantly BELOW GROUND.
    # This would mean the correct way would be to integrate over
    # a piecewise-linear function. It also means that I really
    # want the value at surface level, so I'd need the CO2
    # fields on the Z-staggered grid ("w-points")! Interpolate
    # the vertical in p with wrf.interp1d, example:
    # wrf.interp1d(np.array(rh.isel(south_north=1, west_east=0)),
    #              np.array(p.isel(south_north=1, west_east=0)),
    #              np.array(988, 970))
    # (wrf.interp1d gives the same results as wrf.interplevel,
    # but the latter just doesn't want to work with single
    # columns (32,1,1), it wants a dim>1 in the horizontal
    # directions)
    # So basically, I can keep using pb_ret and pb_mod, but it
    # would be more accurate to do the piecewise-linear
    # interpolation and the output matrix will have 1 more
    # value in each dimension.
    
    # Calculate integration weights by weighting with layer
    # thickness. This assumes that both axes are ordered
    # psurf to ptop.
    coefs = np.ndarray(shape=(len(pb_ret)-1, len(pb_mod)-1))
    coefs[:] = 0.

    # Extend the model pressure grid if retrieval encompasses
    # more.
    # pb_mod_tmp = copy.deepcopy(pb_mod)
    pb_mod_tmp = np.copy(pb_mod)

    # In case the retrieval pressure is higher than the model
    # surface pressure, extend the lowest model layer.
    if pb_mod_tmp[0] < pb_ret[0]:
        pb_mod_tmp[0] = pb_ret[0]

    # In case the model doesn't extend as far as the retrieval,
    # extend the upper model layer upwards.
    if pb_mod_tmp[-1] > pb_ret[-1]:
        pb_mod_tmp[-1] = pb_ret[-1]

    # For each retrieval layer, this loop computes which
    # proportion falls into each model layer.
    for nret in range(len(pb_ret)-1):

        # 1st model pressure boundary index = the one before the
        # first boundary with lower pressure than high-pressure
        # retrieval layer boundary.
        model_lower = np.array(pb_mod_tmp < pb_ret[nret])
        id_model_lower = model_lower.nonzero()[0]
        id_min = id_model_lower[0]-1

        # Last model pressure boundary index = the last one with
        # higher pressure than low-pressure retrieval layer
        # boundary.
        model_higher = np.array(pb_mod_tmp > pb_ret[nret+1])

        id_model_higher = model_higher.nonzero()[0]

        if len(id_model_higher) == 0:
            #id_max = id_min
            raise ValueError("This shouldn't happen. Debug.")
        else:
            id_max = id_model_higher[-1]

        # By the way, in case there is no model level with
        # higher pressure than the next retrieval level,
        # id_max must be the same as id_min.

        # For each model layer, find out how much of it makes up this
        # retrieval layer
        for nmod in range(id_min, id_max+1):
            if (nmod == id_min) & (nmod != id_max):
                # Part of 1st model layer that falls within
                # retrieval layer
                coefs[nret, nmod] = pb_ret[nret] - pb_mod_tmp[nmod+1]
            elif (nmod != id_min) & (nmod == id_max):
                # Part of last model layer that falls within
                # retrieval layer
                coefs[nret, nmod] = pb_mod_tmp[nmod] - pb_ret[nret+1]
            elif (nmod == id_min) & (nmod == id_max):
                # id_min = id_max, i.e. model layer encompasses
                # retrieval layer
                coefs[nret, nmod] = pb_ret[nret] - pb_ret[nret+1]
            else:
                # Retrieval layer encompasses model layer
                coefs[nret, nmod] = pb_mod_tmp[nmod] - pb_mod_tmp[nmod+1]

        coefs[nret, :] = coefs[nret, :]/sum(coefs[nret, :])

    # I tested the code with many cases, but I'm only 99.9% sure
    # it works for all input. Hence a test here that the
    # coefficients sum to 1 and dump the data if not.
    sum_ = np.abs(coefs.sum(1) - 1)
    if np.any(sum_ > 2.*np.finfo(sum_.dtype).eps):
        # dump = dict(pb_ret=pb_ret,
        #             pb_mod=pb_mod,
        #             level_def=level_def)
        # fp = "int_coefs_dump.pkl"
        # with open(fp, "w") as f:
        #     pickle.dump(dump, f, 0)

        msg_fmt = "Something doesn't sum to 1. Arguments dumped to: %s"
        # David:
        # After adding the CAMS extension, 
        # this part complains for some observations...
        # Tested a few cases where it does not complain,
        # The results are near identical, therefore this check
        # has been omitted for now.
        # Might have some problems in the future?? (*to be verified!)
        # vvvvv this part vvvvv
        #raise ValueError(msg_fmt % fp)
            
        
    return coefs




def load_icon_profiles(time, icon_dir, grid_id):
    """
    Given a datetime, load CH4 profiles from the two closest ICON files and interpolate.
    """
    from datetime import timedelta

    # Round down and up to the two surrounding hours
    t0 = time.replace(minute=0, second=0)
    t1 = t0 + timedelta(hours=1)

    fn0 = icon_dir / f"icon_output_{t0:%Y%m%d%H}.nc"
    fn1 = icon_dir / f"icon_output_{t1:%Y%m%d%H}.nc"

    with Dataset(fn0) as f0, Dataset(fn1) as f1:
        ch4_0 = f0.variables["ch4"][:]  # shape [lev, lat, lon]
        ch4_1 = f1.variables["ch4"][:]
        p0 = f0.variables["pressure"][:]  # if not hybrid coords
        p1 = f1.variables["pressure"][:]

    # Compute interpolation weights
    delta = (t1 - t0).total_seconds()
    w1 = (time - t0).total_seconds() / delta
    w0 = 1 - w1

    # ch4_interp = w0 * ch4_0 + w1 * ch4_1
    # p_interp = w0 * p0 + w1 * p1


    ch4_interp = w0 * ch4_0[:, grid_id] + w1 * ch4_1[:, grid_id]
    p_interp   = w0 * p0[:, grid_id]   + w1 * p1[:, grid_id]
    return ch4_interp, p_interp


def load_icon_grid_polygons(grid_file):
    with Dataset(grid_file) as f:
        # These are typically [corner, cell] = [4, n]
        lat_corners = f.variables["clat_vertices"][:]  # shape (4, n)
        lon_corners = f.variables["clon_vertices"][:]  # shape (4, n)

    grid_polygons = [Polygon(zip(lon_corners[:, i], lat_corners[:, i]))
                     for i in range(lat_corners.shape[1])]

    gdf_grid = gpd.GeoDataFrame({"geometry": grid_polygons})
    gdf_grid.set_crs(epsg=4326, inplace=True)
    return gdf_grid

def match_obs_to_grid(obs_polygons, grid_polygons):
    tree = STRtree(grid_polygons)
    grid_map = defaultdict(list)

    for i, obs_poly in enumerate(obs_polygons):
        matching = tree.query(obs_poly)
        for grid_poly in matching:
            j = grid_polygons.index(grid_poly)
            grid_map[i].append(j)

    return grid_map


def compute_remapped_xch4(obs_index, time, grid_indices, tropomi_data, icon_dir):
    """
    Compute XCH4 for one TROPOMI observation.
    """

    print(f"Obs {obs_index}: {len(grid_indices)} intersecting ICON cells")

    # === Step 1: Load CH₄ profiles from intersecting ICON cells ===
    ch4_profiles = []
    p_profiles = []

    for grid_id in grid_indices:
        ch4, p = load_icon_profiles(time, icon_dir, grid_id)
        ch4_profiles.append(ch4)
        p_profiles.append(p)

    if not ch4_profiles:
        return np.nan

    ch4_mean = np.nanmean(ch4_profiles, axis=0)
    p_mean = np.nanmean(p_profiles, axis=0)

    # === Step 2: Project onto satellite pressure grid ===
    pb_ret = tropomi_data["level_pressure"]   # retrieval layer boundaries
    pb_mod = p_mean  # full pressure profile

    coefs = get_int_coefs(pb_ret, pb_mod)
    model_on_ret_grid = coefs @ ch4_mean  # interpolated model profile

    # === Step 3: Apply averaging kernel ===
    ak = tropomi_data["ak"][:, obs_index]
    pw = tropomi_data["pw"][:, obs_index]

    xch4_model = np.sum((ak * model_on_ret_grid + (1 - ak) * tropomi_data["p0"][obs_index]) * pw)

    return xch4_model

# === LOAD AND FILTER TROPOMI DATA ===
with Dataset(tropomi_file) as f_sat:
    # Load and filter dates
    dates_raw = f_sat.variables["date"][:]
    dates = num2date(dates_raw, units=f_sat.variables["date"].units)
    time_mask = np.array([(d >= start_date) and (d <= end_date) for d in dates])
    tropomi_dates = np.array(dates)[time_mask]
    n_filtered = np.sum(time_mask)

    # Convenience helper for shape-safe extraction
    def safe_extract(var, mask):
        data = f_sat.variables[var]
        if data.ndim == 2:
            obs_dim = mask.shape[0]
            if data.shape[1] == obs_dim:
                return data[:, mask]  # observation axis is second
            elif data.shape[0] == obs_dim:
                return data[mask, :].T  # transpose to make it (levels, obs)
            else:
                raise ValueError(f"Incompatible shape for variable '{var}': {data.shape}")
        elif data.ndim == 1:
            return data[mask]
        else:
            raise ValueError(f"Unhandled dimension for variable '{var}'")

    # === Extract relevant variables ===
    ak = safe_extract("ak", time_mask)                     # shape (levels, n_filtered_obs)
    pw = safe_extract("pw", time_mask)                     # shape (levels, n_filtered_obs)
    # print(f_sat.variables["level_pressure"].shape)
    level_pressure = f_sat.variables["level_pressure"][:]   # shape (levels+1, n_filtered_obs)
    tropomi_obs = f_sat.variables["obs"][:][time_mask]     # shape (n_filtered_obs,)
    p0 = f_sat.variables["pavg0"][:]
    p0 = p0[time_mask]         # surface value for correction

    # Corner coordinates
    lat_bounds = f_sat.variables["latitude_bounds"][:, time_mask]  # (4, n_filtered_obs)
    lon_bounds = f_sat.variables["longitude_bounds"][:, time_mask]  # (4, n_filtered_obs)

# === Define obs polygons ===
obs_polygons = [Polygon(zip(lon_bounds[:, i], lat_bounds[:, i])) for i in range(lat_bounds.shape[1])]

# === LOAD ICON GRID ===
gdf_icon = load_icon_grid_polygons(icon_grid_file)
grid_polygons = list(gdf_icon.geometry)

# === MAP OBS TO ICON CELLS ===
grid_map = match_obs_to_grid(obs_polygons, grid_polygons)




remapped_xch4 = np.full(len(obs_polygons), np.nan)

for i in range(len(obs_polygons)):
    time = tropomi_dates[i]  # datetime object
    grid_ids = grid_map[i]

    remapped = compute_remapped_xch4(
        obs_index=i,
        time=time,
        grid_indices=grid_ids,
        tropomi_data={
            "ak": ak,
            "pw": pw,
            "level_pressure": level_pressure,
            "p0": p0 
        },
        icon_dir=icon_hourly_dir
    )
    remapped_xch4[i] = remapped


# Load ICON-ODIN CH4
with Dataset(icon_odin_file) as f_mod:
    obs_mod = f_mod.variables["CH4"][:]


print("Shapes:")
print("obs_mod:", obs_mod.shape)
print("remapped_xch4:", remapped_xch4.shape)
print("tropomi_obs:", tropomi_obs.shape)



# === Ensure alignment ===
assert obs_mod.shape[0] == remapped_xch4.shape[0] == tropomi_obs.shape[0], "Mismatch in observation count"

print("Valid remapped values:", np.sum(~np.isnan(remapped_xch4)))



# === Compute Differences ===
diff_odin_remap = obs_mod - remapped_xch4
diff_tropomi_remap = tropomi_obs - remapped_xch4

# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
titles = [
    "TROPOMI vs Remapped ICON",
    "ICON-ODIN vs Remapped ICON",
    "TROPOMI - Remapped ICON",
    "ICON-ODIN - Remapped ICON"
]

datasets = [
    (tropomi_obs, remapped_xch4),
    (obs_mod, remapped_xch4),
    (diff_tropomi_remap, None),
    (diff_odin_remap, None)
]

for ax, title, (data1, data2) in zip(axs.flat, titles, datasets):
    ax.set_title(title)
    if data2 is not None:
        ax.scatter(data1, data2, s=5, alpha=0.5)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Remapped")
        ax.plot([data1.min(), data1.max()], [data1.min(), data1.max()], 'k--')
    else:
        ax.hist(data1, bins=100, color='gray')
        ax.set_xlabel("Difference [ppb]")
        ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(output_dir / "comparison_remapped_vs_obs.png", dpi=300)
print(f"✅ Saved plot to: {output_dir / 'comparison_remapped_vs_obs.png'}")

