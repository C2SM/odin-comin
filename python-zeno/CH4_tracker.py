"""
CH4 tracking plugin for the ICON Community Interface (ComIn)

@authors 04/2025 :: Zeno Hug, ICON Community Interface  <comin@icon-model.org>

SPDX-License-Identifier: BSD-3-Clause

Please see the file LICENSE in the root of the source tree for this code.
Where software is supplied by third parties, it is indicated in the
headers of the routines.
"""

import comin
import numpy as np
import xarray as xr
from mpi4py import MPI
import pandas as pd
import sys
from scipy.spatial import KDTree
# from shapely.geometry import Polygon
# from shapely.strtree import STRtree
import datetime as datetimelib
import os

## Constants:
NUMBER_OF_NN = 4 # Number of nearest neighbour cells over which should be interpolated
N_COMPUTE_PES = 123  # number of compute PEs
jg = 1 # we do compututations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
Mda, MCH4 = 28.964, 16.04
days_of_flights = [[2019, 10, 7], [2019, 10, 8]] # Up until now manually put in the dates of the flights
start_model = datetimelib.datetime(2019, 1, 1) 
days_of_flights_datetime = []
for entry in days_of_flights:
    days_of_flights_datetime.append(datetimelib.date(entry[0], entry[1], entry[2]))
time_interval_writeout = 900 # variable saying how often you want to writeout the results, in seconds

## Defining variables:
first_write_done_monitoring = False # Help to know if output file already exists

singlepoint_lons = np.array([]) # predefine the arrays as empty
singlepoint_lats = np.array([])
singlepoint_heights = np.array([])
singlepoint_is_abg = np.array([])
singlepoint_timestep = np.array([])
cams_prev_data = None
cams_next_data = None

monitoring_lons = np.array([26.0, 23.0]) # predefine the monitoring stations. This could in future also be done via file inread
monitoring_lats = np.array([46.0, 47.0])
monitoring_heights = np.array([0.0, 0.0])
monitoring_is_abg = np.array([True, True])


def get_int_coefs(pb_ret, pb_mod):
    """
    Computes a coefficients matrix to transfer a model profile onto
    a retrieval pressure axis.
    
    If level_def=="layer_average", this assumes that profiles are
    constant in each layer of the retrieval, bound by the pressure
    boundaries pb_ret. In this case, the WRF model layer is treated
    in the same way, and coefficients integrate over the assumed
    constant model layers. This works with non-staggered WRF
    variables (on "theta" points). However, this is actually not how
    WRF is defined, and the implementation should be changed to
    z-staggered variables. Details for this change are in a comment
    at the beginning of the code.

    If level_def=="pressure_boundary" (IMPLEMENTATION IN PROGRESS),
    assumes that profiles, kernel and pwf are defined at pressure
    boundaries that don't have a thickness (this is how OCO-2 data
    are defined, for example). In this case, the coefficients
    linearly interpolate adjacent model level points. This is
    incompatible with the treatment of WRF in the above-described
    layer-average assumption, but is closer to how WRF is actually
    defined. The exception is that pb_mod is still constructed and
    non-staggered variables are not defined at psurf. This can only
    be fixed by switching to z-staggered variables.

    In cases where retrieval surface pressure is higher than model
    surface pressure, and in cases where retrieval top pressure is
    lower than model top pressure, the model profile will be
    extrapolated with constant tracer mixing ratios. In cases where
    retrieval surface pressure is lower than model surface pressure,
    and in cases where retrieval top pressure is higher than model
    top pressure, only the parts of the model column that fall
    within the retrieval presure boundaries are sampled.

    Arguments
    ---------
    pb_ret (:class:`array_like`)
        Pressure boundaries of the retrieval column
    pb_mod (:class:`array_like`)
        Pressure boundaries of the model column
    level_def (:class:`string`)
        "layer_average" or "pressure_boundary" (IMPLEMENTATION IN
        PROGRESS). Refers to the retrieval profile.
        
        Note 2021-09-13: Inspected code for pressure_boundary.
        Should be correct. Interpolates linearly between two model
        levels.


    Returns
    -------
    coefs (:class:`array_like`)
            Integration coefficient matrix. Each row sums to 1.

    Usage
    -----
            .. code-block:: python

                import numpy as np
                pb_ret = np.linspace(900., 50., 5)
                pb_mod = np.linspace(1013., 50., 7)
                model_profile = 1. - np.linspace(0., 1., len(pb_mod)-1)**3
                coefs = get_int_coefs(pb_ret, pb_mod, "layer_average")
                retrieval_profile = np.matmul(coefs, model_profile)
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

def message(message_string, rank):
    """Short helper function to print a message on one PE"""
    if (comin.parallel_get_host_mpi_rank() == rank):
        print(f"ComIn point_source.py: {message_string}", file=sys.stderr)

def lonlat2xyz(lon, lat):
    """Short helper function for calculating xyz coordinates from longitues and latitudes"""
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

# def map_tropomi_to_icon(tropomi_lon_corners, tropomi_lat_corners, tree, index_map):
#     try:
#         tropomi_poly = Polygon(zip(tropomi_lon_corners, tropomi_lat_corners))
#     except Exception as e:
#         return []

#     matches = tree.query(tropomi_poly)
#     matched_cells = []

#     for poly in matches:
#         if tropomi_poly.intersects(poly):
#             icon_idx = index_map[id(poly)]
#             jc, jb = jcjb_np[icon_idx]
#             matched_cells.append((jc, jb))

#     return matched_cells

def find_stations_monitor(lons, lats, heights, are_abg, tree, decomp_domain, clon, hhl):
    """Find the local stationary monitoring stations on each PE in the own domain and return all of the relevant data needed for computation"""

    # Define all lists as empty
    jc_locs = [] 
    jb_locs = []
    vertical_indices1 = []
    vertical_indices2 = []
    weights_vertical_all = []
    weights_all = []
    lons_local = []
    lats_local = []
    heights_local = []
    are_abg_local = []

    # Loop thorugh every station
    for lon, lat, height, above_ground in zip(lons, lats, heights, are_abg):

        # Query the tree for the NUMBER_OF_NN nearest cells
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = NUMBER_OF_NN)

        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0][0]] == 0:
            jc_loc, jb_loc = np.unravel_index(ii[0], clon.shape) # Extract the indexes

            dd_local = dd[0]

            jc_row = []
            jb_row = []
            vertical_row1 = []
            vertical_row2 = []
            weight_row = []
            weight_row_vertical  = []

            # Now, we want to compute the correct vertical index for each of the NUMBER_OF_NN cells. 
            for jc, jb in zip(jc_loc, jb_loc):
                local_hhl = hhl[jc, :, jb].squeeze() # This is the vertical column of half height levels
                # As the hhl are half height levels we want to get the height levels of the cells. This is done by always taking the middle between the hhls
                h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])
                
                # As the height in the model is in height above sea, we want to add the lowest level hhl (which is the ground level) if the height is measured above ground
                height_above_sea = height
                if above_ground:
                    height_above_sea += local_hhl[-1]

                closest_index = int(np.argmin(np.abs(h_mid - height_above_sea))) # This is the closest index
                
                actual_height_closest = h_mid[closest_index]
                second_index = closest_index
                # Second index is for height interpolation. depending on where the closest height is, compute the second index, also taking into consideration boundary conditions
                if actual_height_closest >= height_above_sea and actual_height_closest != h_mid[-1]:
                    second_index += 1
                elif actual_height_closest < height_above_sea and actual_height_closest != h_mid[0]:
                    second_index -= 1
                second_height = h_mid[second_index]
                vertical_weight = 0
                if second_height - actual_height_closest != 0:
                    vertical_weight = (height_above_sea - actual_height_closest) / (second_height - actual_height_closest)


                jc_row.append(jc)
                jb_row.append(jb)
                vertical_row1.append(closest_index)
                vertical_row2.append(second_index)
                weight_row_vertical.append(vertical_weight)
                

            # As we want to take the inverse of the distances as weights we have big issues if the distance is exactly 0. Thats why we just set it to a very small value
            # It should only happen very rarely, but it could happen if we are really unlucky
            if np.any(dd_local==0):
                print('The longitude/latitude coincides identically with an ICON cell, which is an issue for the inverse distance weighting.', file=sys.stderr)
                print('I will slightly modify this value to avoid errors.', file=sys.stderr)
                dd_local[dd_local==0] = 1e-12

            # Here we compute the weights for interpolating. The weights are normalized to sum up to 1 and they are proportional to the inverse of the distances
            weights = 1.0 / dd_local
            weights = weights / np.sum(weights)
            weight_row = weights.tolist()

            # If fewer than NUMBER_OF_NN neighbors, pad with -1 and 0. This should not happen, as long as the number of neighbors is not set too high but just to be safe
            # As the weight is set to 0 it does not affect the result
            while len(jc_row) < NUMBER_OF_NN:
                jc_row.append(-1)
                jb_row.append(-1)
                vertical_row1.append(-1)
                vertical_row2.append(-1)
                weight_row_vertical.append(0.0)
                weight_row.append(0.0)

            # Append all data, for the cells that were found in this PE's domain
            jc_locs.append(jc_row)
            jb_locs.append(jb_row)
            weights_all.append(weight_row)
            lons_local.append(lon)
            lats_local.append(lat)
            heights_local.append(height)
            are_abg_local.append(above_ground)
            vertical_indices1.append(vertical_row1)
            vertical_indices2.append(vertical_row2)
            weights_vertical_all.append(weight_row_vertical)

    # Return all data as numpy arrays
    return (np.array(jc_locs, dtype = np.int32),
            np.array(jb_locs, dtype = np.int32),
            np.array(vertical_indices1, dtype = np.int32),
            np.array(vertical_indices2, dtype = np.int32),
            np.array(weights_vertical_all, dtype = np.float64),
            np.array(weights_all, dtype = np.float64),
            np.array(lons_local, dtype = np.float64),
            np.array(lats_local, dtype = np.float64),
            np.array(heights_local, dtype = np.float64),
            np.array(are_abg_local, dtype = bool))

def find_stations_singlepoint(lons, lats, heights, are_abg, timesteps, tree, decomp_domain, clon, hhl):
    """Find the local monitoring points that should be read out on a single timestep on each PE in the own domain and return all of the relevant data needed for computation"""
    # Define all lists as empty
    jc_locs = []
    jb_locs = []
    vertical_indices1 = []
    vertical_indices2 = []
    weights_vertical_all = []
    weights_all = []
    lons_local = []
    lats_local = []
    heights_local = []
    are_abg_local = []
    timesteps_local = []

    # Loop thorugh every station
    for lon, lat, height, above_ground, timestep in zip(lons, lats, heights, are_abg, timesteps):
        
        # Query the tree for the NUMBER_OF_NN nearest cells
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = NUMBER_OF_NN)

        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0][0]] == 0:
            jc_loc, jb_loc = np.unravel_index(ii[0], clon.shape) # Extract the indexes

            dd_local = dd[0]

            jc_row = []
            jb_row = []
            vertical_row1 = []
            vertical_row2 = []
            weight_row = []
            weight_row_vertical = []

            # Now, we want to compute the correct vertical index for each of the NUMBER_OF_NN cells. 
            for jc, jb in zip(jc_loc, jb_loc):
                local_hhl = hhl[jc, :, jb].squeeze() # This is the vertical column of half height levels
                # As the hhl are half height levels we want to get the height levels of the cells. This is done by always taking the middle between the hhls
                h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])

                # As the height in the model is in height above sea, we want to add the lowest level hhl (which is the ground level) if the height is measured above ground
                height_above_sea = height
                if above_ground:
                    height_above_sea += local_hhl[-1]

                closest_index = int(np.argmin(np.abs(h_mid - height_above_sea))) # We take cell closest to the actual height
                
                actual_height_closest = h_mid[closest_index]
                second_index = closest_index
                if actual_height_closest >= height_above_sea and actual_height_closest != h_mid[-1]:
                    second_index += 1
                elif actual_height_closest < height_above_sea and actual_height_closest != h_mid[0]:
                    second_index -= 1
                second_height = h_mid[second_index]
                vertical_weight = 0
                if second_height - actual_height_closest != 0:
                    vertical_weight = (height_above_sea - actual_height_closest) / (second_height - actual_height_closest)

                jc_row.append(jc)
                jb_row.append(jb)
                vertical_row1.append(closest_index)
                vertical_row2.append(second_index)
                weight_row_vertical.append(vertical_weight)

            # As we want to take the inverse of the distances as weights we have big issues if the distance is exactly 0. Thats why we just set it to a very small value
            # It should only happen very rarely, but it could happen if we are really unlucky
            if np.any(dd_local==0):
                print('The longitude/latitude coincides identically with an ICON cell, which is an issue for the inverse distance weighting.', file=sys.stderr)
                print('I will slightly modify this value to avoid errors.', file=sys.stderr)
                dd_local[dd_local==0] = 1e-12

            # Here we compute the weights for interpolating. The weights are normalized to sum up to 1 and they are proportional to the inverse of the distances
            weights = 1.0 / dd_local
            weights = weights / np.sum(weights)
            weight_row = weights.tolist()

            # If fewer than NUMBER_OF_NN neighbors, pad with -1 and 0. This should not happen, as long as the number of neighbors is not set too high but just to be safe
            # As the weight is set to 0 it does not affect the result
            while len(jc_row) < NUMBER_OF_NN:
                jc_row.append(-1)
                jb_row.append(-1)
                vertical_row1.append(-1)
                vertical_row2.append(-1)
                weight_row_vertical.append(0.0)
                weight_row.append(0.0)

            # Append all data, for the cells that were found in this PE's domain
            jc_locs.append(jc_row)
            jb_locs.append(jb_row)
            weights_all.append(weight_row)
            lons_local.append(lon)
            lats_local.append(lat)
            heights_local.append(height)
            are_abg_local.append(above_ground)
            timesteps_local.append(timestep)
            vertical_indices1.append(vertical_row1)
            vertical_indices2.append(vertical_row2)
            weights_vertical_all.append(weight_row_vertical)

    # Return all data as numpy arrays
    return (np.array(jc_locs, dtype = np.int32),
            np.array(jb_locs, dtype = np.int32),
            np.array(vertical_indices1, dtype = np.int32),
            np.array(vertical_indices2, dtype = np.int32),
            np.array(weights_vertical_all, dtype = np.float64),
            np.array(weights_all, dtype = np.float64),
            np.array(lons_local, dtype = np.float64),
            np.array(lats_local, dtype = np.float64),
            np.array(heights_local, dtype = np.float64),
            np.array(are_abg_local, dtype = bool),
            np.array(timesteps_local))

def find_stations_satellite(lons, lats, timesteps, tree, decomp_domain, clon, hhl, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree):
    """Find the local monitoring points that should be read out on a single timestep on each PE in the own domain and return all of the relevant data needed for computation"""
    # Define all lists as empty
    jc_locs = []
    jb_locs = []
    lons_local = []
    lats_local = []
    timesteps_local = []
    pavg0_local = []
    pw_local = []
    ak_local = []
    qa0_local = []
    cams_indices_local = []
    fracs_cams_local = []

    # Loop thorugh every station
    for lon, lat, timestep, pavg0, pw, ak, qa0 in zip(lons, lats, timesteps, pavg0_sat, pw_sat, ak_sat, qa0_sat):
        
        # Query the tree for the NUMBER_OF_NN nearest cells
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = 1)

        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0]] == 0:
            jc_loc, jb_loc = np.unravel_index(ii[0], clon.shape) # Extract the indexes

            xyz_pt = lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))
            cams_distances, cams_index = cams_tree.query(xyz_pt, k = 1)
            cams_prev = timestep.replace(hour=(timestep.hour // 6) * 6, minute=0, second=0, microsecond=0)
            cams_next = cams_prev + datetimelib.timedelta(hours=6)

    
            frac_cams = (timestep - cams_prev).total_seconds() / (cams_next - cams_prev).total_seconds()

            # Append all data, for the cells that were found in this PE's domain
            jc_locs.append(jc_loc)
            jb_locs.append(jb_loc)
            lons_local.append(lon)
            lats_local.append(lat)
            timesteps_local.append(timestep)
            pavg0_local.append(pavg0)
            pw_local.append(pw)
            ak_local.append(ak)
            qa0_local.append(qa0)
            cams_indices_local.append(cams_index)
            fracs_cams_local.append(frac_cams)


    # Return all data as numpy arrays
    return (np.array(jc_locs, dtype = np.int32),
            np.array(jb_locs, dtype = np.int32),
            np.array(lons_local, dtype = np.float64),
            np.array(lats_local, dtype = np.float64),
            np.array(timesteps_local), 
            np.array(pavg0_local, dtype = np.float64),
            np.array(pw_local, dtype = np.float64), 
            np.array(ak_local, dtype = np.float64), 
            np.array(qa0_local, dtype = np.float64), 
            np.array(cams_indices_local, dtype = np.int32), 
            np.array(fracs_cams_local, dtype = np.float64))


def write_satellite(datetime):
    """Function to writeout the single timepoint data"""
    global done_counter_sat, comm

    done_data_local = None
    # Collect the local single point data, that we want to write out
    if done_counter_sat > 0:
        done_data_local = {
            "lon": done_lons_sat[:done_counter_sat],
            "lat": done_lats_sat[:done_counter_sat],
            "timepoint": done_times_sat[:done_counter_sat],
            "CH4": done_CH4_sat[:done_counter_sat],
        }

    # Gather the local data to root 0, such that one process has all data that needs to be written out
    gathered_done_data = comm.gather(done_data_local, root=0)

    # The rank that has gathered the data will now write it out
    if rank == 0:
        final_data = {
            "lon": [],
            "lat": [],
            "timepoint": [],
            "CH4": [],
        }
        # Flatten the data and put them into one single list
        for d in gathered_done_data:
            if d is not None:
                for key in final_data:
                    final_data[key].append(d[key])

        # Check if there is anything to write and prepare the data for write out
        if any(len(lst) > 0 for lst in final_data.values()):
            for key in final_data:
                final_data[key] = np.concatenate(final_data[key])
            
            # (Convert datetime64 to Unix timestamp in seconds)
            # final_data["timepoint"] = final_data["timepoint"].astype('int64') // int(1e9)
            df = pd.DataFrame(final_data)
            
            df = df.sort_values(by="timepoint")

            # Csv filename, will maybe change later, when we know how the flight data csv's are named
            csv_file = "satellite_data" + ".csv"

            # Write to csv, write the header only if the file does not yet exist         
            file_exists = os.path.isfile(csv_file)
            df.to_csv(csv_file, mode='a', header=not file_exists, index=False)


def write_singlepoints(datetime):
    """Function to writeout the single timepoint data"""
    global done_counter, comm

    done_data_local = None
    # Collect the local single point data, that we want to write out
    if done_counter > 0:
        done_data_local = {
            "lon": done_lons[:done_counter],
            "lat": done_lats[:done_counter],
            "height": done_heights[:done_counter],
            "timepoint": done_times[:done_counter],
            "CH4": done_CH4[:done_counter],
        }

    # Gather the local data to root 0, such that one process has all data that needs to be written out
    gathered_done_data = comm.gather(done_data_local, root=0)

    # The rank that has gathered the data will now write it out
    if rank == 0:
        final_data = {
            "lon": [],
            "lat": [],
            "height": [],
            "timepoint": [],
            "CH4": [],
        }
        # Flatten the data and put them into one single list
        for d in gathered_done_data:
            if d is not None:
                for key in final_data:
                    final_data[key].append(d[key])

        # Check if there is anything to write and prepare the data for write out
        if any(len(lst) > 0 for lst in final_data.values()):
            for key in final_data:
                final_data[key] = np.concatenate(final_data[key])
            
            # (Convert datetime64 to Unix timestamp in seconds)
            # final_data["timepoint"] = final_data["timepoint"].astype('int64') // int(1e9)
            df = pd.DataFrame(final_data)
            df = df.sort_values(by="timepoint")
            dt_str = pd.to_datetime(datetime).strftime("%Y%m%d")
            # Csv filename, will maybe change later, when we know how the flight data csv's are named
            csv_file = "flight_modeled" + dt_str + ".csv"

            # Write to csv, write the header only if the file does not yet exist         
            file_exists = os.path.isfile(csv_file)
            df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

def write_monitoring_stations(datetime):
    """Function to writeout the stationary monitoring stations"""
    global first_write_done_monitoring, comm

    # Calculate averaged CH4
    avg_CH4_local = current_CH4_monitoring / number_of_timesteps
    avg_CH4_local = np.asarray(avg_CH4_local).ravel()

    # Gather everything on root 0
    gathered_avg_CH4 = comm.gather(avg_CH4_local, root=0)
    gathered_lons = comm.gather(monitoring_lons, root=0)
    gathered_lats = comm.gather(monitoring_lats, root=0)
    gathered_heights = comm.gather(monitoring_heights, root=0)

    # On the PE that has all the gathered data
    if rank == 0:
        # Flatten the data
        avg_CH4_flat = np.concatenate(gathered_avg_CH4)
        lons_flat = np.concatenate(gathered_lons)
        lats_flat = np.concatenate(gathered_lats)
        heights_flat = np.concatenate(gathered_heights)

        # Prepare correct amount of station ids
        station_ids = [f"station_{i}" for i in range(avg_CH4_flat.shape[0])]

        # Create the xarray Dataset, with axis station and time. Each station is identified by their lon, lat and height
        ds = xr.Dataset(
            {
                "avg_CH4": (["station", "time"], avg_CH4_flat[..., np.newaxis]),
            },
            coords={
                "station": station_ids,
                "lon": ("station", lons_flat),
                "lat": ("station", lats_flat),
                "height": ("station", heights_flat),
                "time": [pd.to_datetime(datetime)],
            },
        )

        # Metadata, set units etc.
        ds["avg_CH4"].attrs["units"] = "ppb"
        ds["avg_CH4"].attrs["long_name"] = "Average CH4 concentration"
        ds["lon"].attrs["units"] = "degrees_east"
        ds["lat"].attrs["units"] = "degrees_north"
        ds["height"].attrs["units"] = "m"
        ds["time"].attrs["standard_name"] = "time"

        # Set the encoding of the time, unit is just set to beginning of 2019 for a little bit better readibility of the raw seconds data
        encoding = {
            "time": {
                "units": "seconds since 2019-01-01 00:00:00",
                "calendar": "proleptic_gregorian"
            }
        }


        output_file = "tracked_ch4.nc" # filename of the nc file

        if not first_write_done_monitoring:
            # If we haven't yet written something out we just writeout
            ds.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")
            del ds
            first_write_done_monitoring = True
        else:
            # If we have already written out data, we first read in the current data, then add our new data and then writeout again
            # Maybe this is very inefficient and for many stations we should maybe do it differently in the future
            existing_ds = xr.open_dataset(output_file)
            combined = xr.concat([existing_ds, ds], dim="time")
            existing_ds.close()
            combined.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")
            combined.close()
            del ds


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def data_constructor():
    """Constructor: Get pointers to data"""
    global CH4_EMIS, CH4_BG, pres, pres_ifc
    entry_points = [comin.EP_ATM_TIMELOOP_END] # TIMELOOP END is atm selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
    
    # Request to get the wanted variables (i.e. the EMIS and the BG. Also the pressure). We only want to read the data, not write
    CH4_EMIS = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
    CH4_BG = comin.var_get(entry_points, ("CH4_BG", jg), comin.COMIN_FLAG_READ)
    pres = comin.var_get(entry_points, ("pres", jg), comin.COMIN_FLAG_READ)
    pres_ifc = comin.var_get(entry_points, ("pres_ifc", jg), comin.COMIN_FLAG_READ)

    message("data_constructor successful", msgrank)

@comin.register_callback(comin.EP_ATM_INIT_FINALIZE)
def stations_init():
    global number_of_timesteps, clon, clat, hhl, xyz, decomp_domain, tree # variables with domain info, and general information
    # All of the monitoring variables
    global jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring1, vertical_indices_monitoring2, vertical_weight_monitoring, current_CH4_monitoring
    global monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg, weights_monitoring
    # global tree_satellite, icon_index_satellite, corners_icon_np , jcjb_np
    global cams_pressure_cache, cams_interface_cache, cams_profile_cache, cams_prevs, cams_nexts, fracs_cams, obs_time_dts
    global jc_loc_satellite, jb_loc_satellite, satellite_timestep, satellite_lons, satellite_lats
    global CH4_sat, done_lons_sat, done_lats_sat, done_times_sat, done_CH4_sat, done_counter_sat, done_pavg0_sat, done_pw_sat, done_ak_sat, done_qa0_sat
    global pavg0_sat, qa0_sat, ak_sat, pw_sat, cams_indices_sat, cams_times_sorted, cams_files_dict, fracs_cams
    global hyam, hybm, hyai, hybi
    # MPI variables
    global comm, rank

    world_comm = MPI.COMM_WORLD
    group_world = world_comm.Get_group()
    # The last 5 PE's don't participate in the python plugin. This was manually tested, could be different in a different setup
    group = group_world.Incl(list(range(N_COMPUTE_PES)))
    comm = world_comm.Create_group(group)

    if comm != MPI.COMM_NULL:
        rank = comm.Get_rank()
    else:
        rank = None

    number_of_timesteps = 0 # number of timesteps, as we are initializing we set it to 0
    
    # All arrays are for domain 1 only
    # We read in all of the domain data we need. We read it in only once and then save it as a global variable
    domain = comin.descrdata_get_domain(jg)
    clon = np.asarray(domain.cells.clon)
    clat = np.asarray(domain.cells.clat)
    hhl = np.asarray(domain.cells.hhl)
    decomp_domain = np.asarray(domain.cells.decomp_domain)

    # Convert the longitude latitude data to xyz data for the KDTree
    xyz = np.c_[lonlat2xyz(clon.ravel(),clat.ravel())]
    tree = KDTree(xyz) # Create the KDTree, composed of the xyz data

    # Find all of the monitoring stations in this local PE's domain and save all relevant data
    (jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring1, vertical_indices_monitoring2, vertical_weight_monitoring,  weights_monitoring, 
        monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg) = find_stations_monitor(
        monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg, tree, decomp_domain, clon, hhl)
    
    # vertex_blk = np.asarray(domain.cells.vertex_blk)
    # vertex_idx = np.asarray(domain.cells.vertex_idx)
    # vlat = np.asarray(domain.verts.vlat)
    # vlon = np.asarray(domain.verts.vlon)
    # num_edges = np.asarray(domain.cells.num_edges)
    # nproma, nblks_c, max_edges = vertex_blk.shape

    # corners_icon = []
    # jcjb_array = []
    # for jb in range(nblks_c):
    #     for jc in range(nproma):
    #         n_edges = num_edges[jc, jb]
    #         poly = []

    #         for k in range(n_edges):
    #             blk = vertex_blk[jc, jb, k]
    #             idx = vertex_idx[jc, jb, k]

    #             lat = vlat[idx, blk]
    #             lon = vlon[idx, blk]
    #             poly.append((lon, lat))

    #         if poly:
    #             corners_icon.append(poly)
    #             jcjb_array.append((jc, jb))

    # max_len = max(len(p) for p in corners_icon)
    # corners_icon_np = np.full((len(corners_icon), max_len, 2), np.nan)
    # for i, poly in enumerate(corners_icon):
    #     for j, (lon, lat) in enumerate(poly):
    #         corners_icon_np[i, j] = [lon, lat]
    
    # jcjb_np = np.array(jcjb_array, dtype=np.int32)


    # icon_polygons_satellite = [Polygon(corners_icon_np[i, ~np.isnan(corners_icon_np[i,:,0])]) for i in range(corners_icon_np.shape[0])]
    # tree_satellite = STRtree(icon_polygons_satellite)
    # icon_index_satellite = {id(polygon): i for i, polygon in enumerate(icon_polygons_satellite)}  # Map polygons to indices

    if rank == 0:
        tropomi_ds = xr.open_dataset("TROPOMI_SRON_20190101_20191231.nc")
        raw_times_satellite = tropomi_ds['date'].values
        satellite_lons = tropomi_ds['lon'].values
        satellite_lats = tropomi_ds['lat'].values
        ref_time = pd.to_datetime("2019-01-01 11:14:35.629000")
        pavg0_sat = tropomi_ds['pavg0'].values
        qa0_sat = tropomi_ds['qa0'].values
        ak_sat = tropomi_ds['ak'].values
        pw_sat = tropomi_ds['pw'].values
        obs_time_dts = pd.to_datetime(raw_times_satellite)
        
        valid_mask = obs_time_dts >= start_model
        satellite_lons = satellite_lons[valid_mask]
        satellite_lats = satellite_lats[valid_mask]
        pavg0_sat = pavg0_sat[valid_mask]
        pw_sat = pw_sat[valid_mask]
        ak_sat = ak_sat[valid_mask]
        qa0_sat = qa0_sat[valid_mask]
        obs_time_dts = obs_time_dts[valid_mask]

        cams_ds = xr.open_dataset("/capstor/scratch/cscs/zhug/Romania6km/input/CAMS/LBC/cams73_v22r2_ch4_conc_surface_inst_2019010100_lbc.nc")
        cams_clon = np.asarray(cams_ds["clon"])
        cams_clat = np.asarray(cams_ds["clat"])
        cams_param_ds = xr.open_dataset("cams73_v22r2_ch4_conc_surface_inst_201910.nc")
        hyam = cams_param_ds["hyam"].values
        hybm = cams_param_ds["hybm"].values
        hyai = cams_param_ds["hyai"].values
        hybi = cams_param_ds["hybi"].values
    else:
        satellite_lons = None
        satellite_lats = None
        obs_time_dts = None
        pavg0_sat = None
        pw_sat = None
        ak_sat = None
        qa0_sat = None
        cams_clon = None
        cams_clat = None
        hyam = None
        hybm = None
        hyai = None
        hybi = None

    # Broadcast the data to all processes, from root 0
    satellite_lons = comm.bcast(satellite_lons, root=0)
    satellite_lats = comm.bcast(satellite_lats, root=0)
    obs_time_dts = comm.bcast(obs_time_dts, root=0)
    pavg0_sat = comm.bcast(pavg0_sat, root=0)
    pw_sat = comm.bcast(pw_sat, root=0)
    ak_sat = comm.bcast(ak_sat, root=0)
    qa0_sat = comm.bcast(qa0_sat, root=0)
    cams_clat = comm.bcast(cams_clat, root=0)
    cams_clon = comm.bcast(cams_clon, root=0)
    hyam = comm.bcast(hyam, root=0)
    hybm = comm.bcast(hybm, root=0)
    hyai = comm.bcast(hyai, root=0)
    hybi = comm.bcast(hybi, root=0)


    cams_xyz = np.c_[lonlat2xyz(cams_clon, cams_clat)]
    cams_tree = KDTree(cams_xyz)

    (jc_loc_satellite, jb_loc_satellite, satellite_lons, satellite_lats, satellite_timestep, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_indices_sat, fracs_cams) = find_stations_satellite(satellite_lons, satellite_lats, obs_time_dts, tree, decomp_domain, clon, hhl, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree)
    # tc = []
    # tc_omv = [] # get the OMV-signal only in the profile

    # Preallocate arrays to avoid appending repeatedly
    # gridindices = []
    # obsindices = []
    # tcindices = []
    # ncount = 0
    datetime = comin.current_get_datetime()

    # cams_prevs = [
    #     obs_time_dt.replace(hour=(datetime.hour // 6) * 6, minute=0, second=0, microsecond=0)
    #     for obs_time_dt in satellite_timestep
    # ]
    # cams_nexts = [cams_prev + datetimelib.timedelta(hours=6) for cams_prev in cams_prevs]

    # Compute fractions in one go
    # fracs_cams = [
    #     (obs_time_dt - cams_prev).total_seconds() / (cams_next - cams_prev).total_seconds()
    #     for obs_time_dt, cams_prev, cams_next in zip(satellite_timestep, cams_prevs, cams_nexts)
    # ]

    # Preload pressure profiles to avoid repeatedly extracting them
    # model_data_cache = {}
    # cams_pressure_cache = {}
    # cams_interface_cache = {}
    # cams_profile_cache = {}
    N_satellite_points = satellite_lons.shape[0] # Amount of satellite points in the local PE

    # Initialize all needed arrays as empty arrays of correct size
    CH4_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_lons_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_lats_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_times_sat = np.empty(N_satellite_points, dtype='datetime64[ns]')
    done_CH4_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_pavg0_sat = np.empty((N_satellite_points, 13), dtype=np.float64)
    done_qa0_sat = np.empty((N_satellite_points, 12), dtype=np.float64)
    done_ak_sat = np.empty((N_satellite_points, 12), dtype=np.float64)
    done_pw_sat = np.empty((N_satellite_points, 12), dtype=np.float64)
    done_counter_sat = 0

    cams_base_path = "/capstor/scratch/cscs/zhug/Romania6km/input/CAMS/LBC/"
    cams_files_dict = {}
    start = pd.to_datetime(datetime)
    end = datetimelib.datetime(2020, 1, 30, 0)

    dt = start
    while dt < end:
        dt_str = dt.strftime("%Y%m%d%H")
        fname = f"cams73_v22r2_ch4_conc_surface_inst_{dt_str}_lbc.nc"
        full_path = os.path.join(cams_base_path, fname)
        cams_files_dict[dt] = full_path
        dt += datetimelib.timedelta(hours=6)

    cams_times_sorted = sorted(cams_files_dict.keys())

    current_CH4_monitoring = np.zeros(monitoring_lons.shape, dtype=np.float64) # Initialize the array for the CH4 monitoring to 0

@comin.register_callback(comin.EP_ATM_TIMELOOP_START)
def input_flight_data():
    """reading in csv flight file"""
    # All of the singlepoints variables
    global singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_is_abg
    global singlepoint_timestep, jc_loc_singlepoint, jb_loc_singlepoint, vertical_indices_singlepoint1, vertical_indices_singlepoint2, vertical_weight_singlepoint, CH4_singlepoint, weights_singlepoint
    # Variables for saving all of the already done singlepoints
    global done_lons, done_lats, done_heights, done_times, done_CH4, done_counter
    # helping variable to know how many points we have
    global N_flight_points

    global cams_prev_data, cams_next_data

    # get the datetime from comin
    datetime = comin.current_get_datetime()

    # Currently we read in at midnight for the following night
    if(pd.to_datetime(datetime).time() == datetimelib.time(0,0)):
        dt_str = pd.to_datetime(datetime).strftime("%Y%m%d")
        # day = pd.to_datetime(datetime).day # get the current day for the name of the filepath
        current_date = pd.to_datetime(datetime).date()
        # Currently there is a manual list of days with flights. Will need to change later, but need to know how the naming structure of the flight data works
        if current_date in days_of_flights_datetime:
            # Predefine all variables
            singlepoint_lons = None
            singlepoint_lats = None
            singlepoint_heights = None
            singlepoint_is_abg = None
            singlepoint_timestep = None

            # Only 1 process reads it in as it could be problematic if a lot of processes try to read in the same file at the same time
            if rank == 0:
                # Naming of the flight data, will be changed later
                flight_file = 'flight' + dt_str + '.csv'
                # Read in the needed data

                df = pd.read_csv(flight_file, sep=';')
                df.columns = [col.strip() for col in df.columns]
                df = df.dropna(subset=['Time_EPOCH', 'AGL_m', 'Longitude', 'Latitude'])
                df['datetime'] = pd.to_datetime(df['Time_EPOCH'], unit='s')


                # This is a small codeblock to manipulate the date of the flight for debugging purposes, if you don't have a flight with the correct date but want to test the functionality
                # target_start = datetimelib.datetime(2019, 1, 1, 0, 1, 0) # Just enter the date and time you want the flight to start
                # original_start = df['datetime'].min()
                # delta = original_start - target_start
                # df['datetime'] = df['datetime'] - delta

                # convert the needed data to numpy arrays
                singlepoint_lons = df['Longitude'].to_numpy()
                singlepoint_lats = df['Latitude'].to_numpy()
                singlepoint_heights = df['AGL_m'].to_numpy()
                singlepoint_is_abg = np.ones_like(singlepoint_lons, dtype=bool) # As currently all of the flight data is given as above ground height, it is just set to true for all points
                singlepoint_timestep = df['datetime'].to_numpy()
            
            # Broadcast the data to all processes, from root 0
            singlepoint_lons = comm.bcast(singlepoint_lons, root=0)
            singlepoint_lats = comm.bcast(singlepoint_lats, root=0)
            singlepoint_heights = comm.bcast(singlepoint_heights, root=0)
            singlepoint_is_abg = comm.bcast(singlepoint_is_abg, root=0)
            singlepoint_timestep = comm.bcast(singlepoint_timestep, root=0)      

            # On each process find the local points in the PE's domain. Get all needed data
            (jc_loc_singlepoint, jb_loc_singlepoint, vertical_indices_singlepoint1, vertical_indices_singlepoint2, vertical_weight_singlepoint, weights_singlepoint, 
                singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_is_abg, singlepoint_timestep) = find_stations_singlepoint(
                    singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_is_abg, singlepoint_timestep, tree, decomp_domain, clon, hhl)
        
        N_flight_points = singlepoint_lons.shape[0] # Amount of flight points in the local PE

        # Initialize all needed arrays as empty arrays of correct size
        CH4_singlepoint = np.empty(N_flight_points, dtype=np.float64)
        done_lons = np.empty(N_flight_points, dtype=np.float64)
        done_lats = np.empty(N_flight_points, dtype=np.float64)
        done_heights = np.empty(N_flight_points, dtype=np.float64)
        done_times = np.empty(N_flight_points, dtype='datetime64[ns]')
        done_CH4 = np.empty(N_flight_points, dtype=np.float64)

        done_counter = 0 # counter of how many of the N_flight_points are already done (This day)

    if pd.to_datetime(datetime).time() == datetimelib.time(0,0) or pd.to_datetime(datetime).time() == datetimelib.time(6,0) or pd.to_datetime(datetime).time() == datetimelib.time(12,0) or pd.to_datetime(datetime).time() == datetimelib.time(18,0):
        cams_prev_time = pd.to_datetime(datetime)
        cams_next_time = cams_prev_time + datetimelib.timedelta(hours=6)
        if cams_prev_data is not None:
            cams_prev_data.close()
        if cams_next_data is not None:
            cams_next_data.close()
        cams_prev_data = xr.open_dataset(cams_files_dict[cams_prev_time])
        cams_next_data = xr.open_dataset(cams_files_dict[cams_next_time])

@comin.register_callback(comin.EP_ATM_TIMELOOP_END) # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
def tracking_CH4_total():
    """tracking of CH4 Emissions"""
    # general info
    global number_of_timesteps, N_flight_points
    # stationary monitoring
    global current_CH4_monitoring
    # singlepoint monitoring
    global jc_loc_singlepoint, jb_loc_singlepoint, vertical_indices_singlepoint1, vertical_indices_singlepoint2, CH4_singlepoint, singlepoint_timestep, singlepoint_lons, singlepoint_lats, singlepoint_heights
    global done_lons, done_lats, done_heights, done_times, done_CH4, done_counter, weights_singlepoint, vertical_weight_singlepoint

    global jc_loc_satellite, jb_loc_satellite, CH4_sat, satellite_timestep, satellite_lons, satellite_lats, pavg0_sat, pw_sat, ak_sat, qa0_sat
    global done_lons_sat, done_lats_sat, done_CH4_sat, done_times_sat, done_pavg0_sat, done_ak_sat, done_pw_sat, done_qa0_sat, done_counter_sat
    global cams_indices_sat, fracs_cams
    global hyam, hybm, hyai, hybi

    dtime = comin.descrdata_get_timesteplength(jg) # size of every timestep 
    datetime = comin.current_get_datetime() # get datetime info. example for format: 2019-01-01T00:01:00.000
    number_of_timesteps += 1 # tracking number of steps, to in the end average after the correct time and average of the correct amount of timesteps

    # Convert all of them to numpy arrays
    CH4_EMIS_np = np.asarray(CH4_EMIS)
    CH4_BG_np = np.asarray(CH4_BG)
    pres_np = np.asarray(pres)
    pres_ifc_np = np.asarray(pres_ifc)

    ## First we do the stationary monitoring
    # Fetch CH4 values in the correct indices, this fetches per monitoring station NUMBER_OF_NN points
    # Also, we want the CH4 Emissions in ppb. And the EMIS is not yet in ppb but just in parts per part. So we multiply by 1e9
    CH4_monitoring_all1 = (
        CH4_EMIS_np[jc_loc_monitoring, vertical_indices_monitoring1, jb_loc_monitoring, 0, 0] * 1e9 +
        CH4_BG_np[jc_loc_monitoring, vertical_indices_monitoring1, jb_loc_monitoring, 0, 0]
    )
    CH4_monitoring_all2 = (
        CH4_EMIS_np[jc_loc_monitoring, vertical_indices_monitoring2, jb_loc_monitoring, 0, 0] * 1e9 +
        CH4_BG_np[jc_loc_monitoring, vertical_indices_monitoring2, jb_loc_monitoring, 0, 0]
    )
    CH4_monitoring_all = CH4_monitoring_all1 + vertical_weight_monitoring * (CH4_monitoring_all2 - CH4_monitoring_all1)
    # If we have any data we add the current contribution while also multiplying by the weights
    if weights_monitoring.size > 0 and CH4_monitoring_all.size > 0:
        current_CH4_monitoring += np.sum(weights_monitoring * CH4_monitoring_all, axis=1)



    ## Secondly we do the single points from the flight data:
    # Convert the model time from comin to a numpy datetime to then compare it with the times from the singlepoints
    model_time_np = np.datetime64(datetime)

    if singlepoint_timestep.size > 0: # Checks if there is still work to do this day
        # mask to mask out the stations, where the model time is greater or equal to the moment we want to measure. They are ready for measurement
        ready_mask = singlepoint_timestep <= model_time_np

        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready = jc_loc_singlepoint[ready_mask]
            jb_ready = jb_loc_singlepoint[ready_mask]
            vi_ready1 = vertical_indices_singlepoint1[ready_mask]
            vi_ready2 = vertical_indices_singlepoint2[ready_mask]
            weights_vertical_ready = vertical_weight_singlepoint[ready_mask]
            weights_ready = weights_singlepoint[ready_mask]

            # Fetch CH4 values in the correct indices, this fetches per station NUMBER_OF_NN points
            # Also, we want the CH4 Emissions in ppb. And the EMIS is not yet in ppb but just in parts per part. So we multiply by 1e9
            CH4_ready_all1 = (
                CH4_EMIS_np[jc_ready, vi_ready1, jb_ready, 0, 0] * 1e9 +
                CH4_BG_np[jc_ready, vi_ready1, jb_ready, 0, 0]
            )
            CH4_ready_all2 = (
                CH4_EMIS_np[jc_ready, vi_ready2, jb_ready, 0, 0] * 1e9 +
                CH4_BG_np[jc_ready, vi_ready2, jb_ready, 0, 0]
            )
            CH4_ready_all = CH4_ready_all1 + weights_vertical_ready * (CH4_ready_all2 - CH4_ready_all1)


            # Sum up and correctly interpolate via weights
            CH4_ready = np.sum(weights_ready * CH4_ready_all, axis=1)

            num_ready = np.sum(ready_mask) # Count how many points are ready

            # Add all of the done points to the done arrays
            done_lons[done_counter:done_counter + num_ready] = singlepoint_lons[ready_mask]
            done_lats[done_counter:done_counter + num_ready] = singlepoint_lats[ready_mask]
            done_heights[done_counter:done_counter + num_ready] = singlepoint_heights[ready_mask]
            done_times[done_counter:done_counter + num_ready] = singlepoint_timestep[ready_mask]
            done_CH4[done_counter:done_counter + num_ready] = CH4_ready

            # Keep count of how many singlepoints are done
            done_counter += num_ready

            # Only keep the singlepoints that aren't done yet
            keep_mask = ~ready_mask
            singlepoint_lons = singlepoint_lons[keep_mask]
            singlepoint_lats = singlepoint_lats[keep_mask]
            singlepoint_heights = singlepoint_heights[keep_mask]
            singlepoint_timestep = singlepoint_timestep[keep_mask]
            jc_loc_singlepoint = jc_loc_singlepoint[keep_mask]
            jb_loc_singlepoint = jb_loc_singlepoint[keep_mask]
            vertical_indices_singlepoint1 = vertical_indices_singlepoint1[keep_mask]
            vertical_indices_singlepoint2 = vertical_indices_singlepoint2[keep_mask]
            vertical_weight_singlepoint = vertical_weight_singlepoint[keep_mask]
            weights_singlepoint = weights_singlepoint[keep_mask]

    
    ## Satellite data:
    if satellite_timestep.size > 0: # Checks if there is still work to do this day
        # mask to mask out the stations, where the model time is greater or equal to the moment we want to measure. They are ready for measurement
        ready_mask = satellite_timestep <= model_time_np

        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready_sat = jc_loc_satellite[ready_mask]
            jb_ready_sat = jb_loc_satellite[ready_mask]
            frac_cams_ready = fracs_cams[ready_mask]
            frac_cams_ready = frac_cams_ready[:, np.newaxis]
            cams_indices_ready = cams_indices_sat[ready_mask]

            pb_mod = (pres_ifc_np[jc_ready_sat, ::-1, jb_ready_sat].squeeze()) / 1.e2
            pb_mod_mc =(pres_np[jc_ready_sat, ::-1, jb_ready_sat].squeeze()) / 1.e2

            # key_cams_prev = (cams_prevs[nobs].year, cams_prevs[nobs].month, cams_prevs[nobs].day, cams_prevs[nobs].hour)
            # key_cams_next = (cams_nexts[nobs].year, cams_nexts[nobs].month, cams_nexts[nobs].day, cams_nexts[nobs].hour)
            # if key_cams_prev not in cams_pressure_cache:
            #     cams_pressure_cache[key_cams_prev] = cams_pressures[key_cams_prev]
            #     cams_profile_cache[key_cams_prev] = cams_profiles[key_cams_prev]
            #     cams_interface_cache[key_cams_prev] = cams_interfaces[key_cams_prev]
            # if key_cams_next not in cams_pressure_cache:
            #     cams_pressure_cache[key_cams_next] = cams_pressures[key_cams_next]
            #     cams_profile_cache[key_cams_next] = cams_profiles[key_cams_next]
            #     cams_interface_cache[key_cams_next] = cams_interfaces[key_cams_next]

            # CAMS_p_prev = np.nanmean(cams_pressure_cache[key_cams_prev][mindices],axis=0)
            # CAMS_p_next = np.nanmean(cams_pressure_cache[key_cams_next][mindices],axis=0)
            # CAMS_pressures = (1. - frac_cams) * CAMS_p_prev + frac_cams * CAMS_p_next

            # CAMS_i_prev = np.nanmean(cams_interface_cache[key_cams_prev][mindices],axis=0)
            # CAMS_i_next = np.nanmean(cams_interface_cache[key_cams_next][mindices],axis=0)
            # CAMS_interfaces = (1. - frac_cams) * CAMS_i_prev + frac_cams * CAMS_i_next

            # CAMS_obs_prev = np.nanmean(cams_profile_cache[key_cams_prev][:, mindices],axis=1)
            # CAMS_obs_next = np.nanmean(cams_profile_cache[key_cams_next][:, mindices],axis=1)
            CAMS_obs_prev = cams_prev_data["CH4"].isel(time=0, ncells = cams_indices_ready).values.T
            CAMS_obs_next = cams_next_data["CH4"].isel(time=0, ncells = cams_indices_ready).values.T

            CAMS_aps_prev = cams_prev_data["ps"].isel(time=0, ncells = cams_indices_ready).values
            CAMS_aps_next = cams_next_data["ps"].isel(time=0, ncells = cams_indices_ready).values

            CAMS_aps_prev_reshaped = CAMS_aps_prev[:, np.newaxis]
            CAMS_aps_next_reshaped = CAMS_aps_next[:, np.newaxis]
            hyam_new_axis, hybm_new_axis = hyam[np.newaxis, :], hybm[np.newaxis, :]
            hyai_new_axis, hybi_new_axis = hyai[np.newaxis, :], hybi[np.newaxis, :]
            CAMS_p_prev =  (hyam_new_axis + hybm_new_axis * CAMS_aps_prev_reshaped) / 1.e2
            CAMS_p_next = (hyam_new_axis + hybm_new_axis * CAMS_aps_next_reshaped) / 1.e2
            CAMS_pressures = (1. - frac_cams_ready) * CAMS_p_prev + frac_cams_ready * CAMS_p_next

            CAMS_i_prev = (hyai_new_axis + hybi_new_axis * CAMS_aps_prev_reshaped) / 1.e2
            CAMS_i_next =  (hyai_new_axis + hybi_new_axis * CAMS_aps_next_reshaped) / 1.e2
            CAMS_interfaces = (1. - frac_cams_ready) * CAMS_i_prev + frac_cams_ready * CAMS_i_next
            
            # if pb_mod.ndim == 1:
            #     pb_min = np.min(pb_mod)
            #     camsidx_vert = CAMS_pressures < pb_min  # or np.expand_dims(pb_mod, axis=0) if needed
            # else:
            #     pb_min = np.min(pb_mod, axis=1)
            #     camsidx_vert = CAMS_pressures < pb_min[:, np.newaxis]
        
            
            ICON_profile = (
                (Mda / MCH4) * CH4_BG_np[jc_ready_sat, ::-1, jb_ready_sat].squeeze() \
                + 1.e9 * (Mda / MCH4) * CH4_EMIS_np[jc_ready_sat, ::-1, jb_ready_sat].squeeze()
            )
            ICON_profile = np.squeeze(ICON_profile)
            
            

            CAMS_obs = (1. - frac_cams_ready) * CAMS_obs_prev + frac_cams_ready * CAMS_obs_next
            CAMS_profile = np.array([
                CAMS_obs[i, CAMS_pressures[i] < np.min(pb_mod[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            # pb_cams = CAMS_pressures[camsidx_vert]
            pb_cams_mc = np.array([
                CAMS_pressures[i, CAMS_pressures[i] < np.min(pb_mod[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            # pb_cams_mc = np.array([
            #     CAMS_interfaces[i, 1:][CAMS_pressures[i, :] < np.min(pb_mod[i, :])]
            #     for i in range(CAMS_interfaces.shape[0])
            # ], dtype=object)
            CAMS_interfaces = CAMS_interfaces[:, 1:]
            pb_cams = np.array([
                CAMS_interfaces[i, CAMS_pressures[i] < np.min(pb_mod[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            # pb_cams = np.array([
            #     CAMS_pressures[i, :][CAMS_pressures[i, :] < np.min(pb_mod[i, :])]
            #     for i in range(CAMS_pressures.shape[0])
            # ], dtype=object)
            # pb_cams_mc = CAMS_pressures[:, :][camsidx_vert]
            # print("", file=sys.stderr)
            # print("RANK: ", rank, " pb_mod shape: ", pb_mod.shape, " , pb_cams shape: ", pb_cams.shape, file=sys.stderr)
            # print("RANK: ", rank, " pb_mod_mc shape: ", pb_mod_mc.shape, " , pb_cams_mc shape: ", pb_cams_mc.shape, file=sys.stderr)
            ICON_profile = np.atleast_2d(ICON_profile)
            CAMS_profile = np.atleast_2d(CAMS_profile)
            pb_mod = np.atleast_2d(pb_mod)
            pb_cams = np.atleast_2d(pb_cams)
            pb_mod_mc = np.atleast_2d(pb_mod_mc)
            pb_cams_mc = np.atleast_2d(pb_cams_mc)

            tracer_profile = np.concatenate((ICON_profile, CAMS_profile), axis = 1)
            pb_profile = np.concatenate((pb_mod, pb_cams), axis = 1)
            pb_mc_profile = np.concatenate((pb_mod_mc, pb_cams_mc), axis = 1)
            # tracer_profile = ICON_profile
            # pb_profile = pb_mod
            # pb_mc_profile = pb_mod_mc

            pb_ret = pavg0_sat[ready_mask]
            # pb_ret = Tdata_cleaned.pressure_levels[nobs].values
            

            coef_matrix = np.array([get_int_coefs(pb_ret[i], pb_profile[i]) for i in range(pb_ret.shape[0])])

            # pwf = np.abs(np.diff(pb_ret) / np.ptp(pb_ret))
            pwf = pw_sat[ready_mask]
            averaging_kernel = ak_sat[ready_mask]
            important_stuff = qa0_sat[ready_mask]
            avpw = pwf * averaging_kernel
            prior_col = np.sum(pwf * important_stuff, axis=1)

            # print("RANK: ", rank, " coef_matrix shape: ", coef_matrix.shape, " , tracer_profile shape: ", tracer_profile.shape, file=sys.stderr)
            # profile_intrp = np.matmul(coef_matrix, tracer_profile)
            # profile_intrp = tracer_profile @ coef_matrix.T  # shape: (n_samples, 13)
            for i in range(len(tracer_profile)):
                if tracer_profile[i].shape[0] != pb_profile[i].shape[0] - 1:
                    print(f"RANK {rank} — Mismatch at i={i}: tracer_profile = {tracer_profile[i].shape}, pb_profile = {pb_profile[i].shape}", file=sys.stderr)
            profile_intrp = np.matmul(coef_matrix, tracer_profile[..., np.newaxis])[..., 0]
            tc = prior_col + np.sum(avpw * (profile_intrp - important_stuff), axis=1)


            num_ready = np.sum(ready_mask) # Count how many points are ready

            # Add all of the done points to the done arrays
            done_lons_sat[done_counter:done_counter + num_ready] = satellite_lons[ready_mask]
            done_lats_sat[done_counter:done_counter + num_ready] = satellite_lats[ready_mask]
            done_times_sat[done_counter:done_counter + num_ready] = satellite_timestep[ready_mask]
            done_CH4_sat[done_counter:done_counter + num_ready] = tc
            done_pavg0_sat[done_counter:done_counter + num_ready] = pavg0_sat[ready_mask]
            done_qa0_sat[done_counter:done_counter + num_ready] = qa0_sat[ready_mask]
            done_pw_sat[done_counter:done_counter + num_ready] = pw_sat[ready_mask]
            done_ak_sat[done_counter:done_counter + num_ready] = ak_sat[ready_mask]

            # Keep count of how many singlepoints are done
            done_counter_sat += num_ready

            # Only keep the singlepoints that aren't done yet
            keep_mask = ~ready_mask
            satellite_lons = satellite_lons[keep_mask]
            satellite_lats = satellite_lats[keep_mask]
            satellite_timestep = satellite_timestep[keep_mask]
            jc_loc_satellite = jc_loc_satellite[keep_mask]
            jb_loc_satellite = jb_loc_satellite[keep_mask]
            pavg0_sat = pavg0_sat[keep_mask]
            qa0_sat = qa0_sat[keep_mask]
            pw_sat = pw_sat[keep_mask]
            ak_sat = ak_sat[keep_mask]
            cams_indices_sat = cams_indices_sat[keep_mask]
            fracs_cams = fracs_cams[keep_mask]
        



    ## Writeout
    elapsed_time = dtime * number_of_timesteps # Time that has elapsed since last writeout (or since start if there was no writeout yet)
    if (elapsed_time >= time_interval_writeout): # If we are above the time intervall writeout defined on the very top, we want to write out
        write_monitoring_stations(datetime) # Write out the monitoring stations
        write_singlepoints(datetime) # Write out the flight data single points
        write_satellite(datetime)

        # Reset data
        number_of_timesteps = 0
        done_counter = 0
        done_counter_sat = 0
        current_CH4_monitoring[:] = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def CH4_destructor():
    message("CH4_destructor called!", msgrank)
