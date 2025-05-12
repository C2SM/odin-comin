"""
CH4 tracking plugin for the ICON Community Interface (ComIn)

@authors 04/2025 :: Zeno Hug, ICON Community Interface  <comin@icon-model.org>

SPDX-License-Identifier: BSD-3-Clause

Please see the file LICENSE in the root of the source tree for this code.
Where software is supplied by third parties, it is indicated in the
headers of the routines.
"""

import numpy as np
import xarray as xr
from mpi4py import MPI
import pandas as pd
import sys
from scipy.spatial import KDTree
# from shapely.geometry import Polygon
# from shapely.strtree import STRtree
import datetime as datetimelib
from get_int_coefs import *
import os

## Constants:
Mda, MCH4 = 28.964, 16.04

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


def find_stations_satellite(lons, lats, timesteps, tree, decomp_domain, clon, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree):
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


def write_satellite(datetime, comm, done_data):
    """Function to writeout the single timepoint data"""
    # global done_counter_sat, comm

    done_data_local = None
    done_counter = done_data['counter']
    # Collect the local single point data, that we want to write out
    if done_counter > 0:
        done_data_local = {
            "lon": done_data['lon'][:done_counter],
            "lat": done_data['lat'][:done_counter],
            "timepoint": done_data['timestep'][:done_counter],
            "CH4": done_data['CH4'][:done_counter],
        }

    # Gather the local data to root 0, such that one process has all data that needs to be written out
    gathered_done_data = comm.gather(done_data_local, root=0)

    # The rank that has gathered the data will now write it out
    if comm.Get_rank() == 0:
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

            # Csv filename, will maybe change later, when we know how the satellite data csv is named
            csv_file = "satellite_data" + ".csv"

            # Write to csv, write the header only if the file does not yet exist         
            file_exists = os.path.isfile(csv_file)
            df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

def read_in_satellite_data(datetime, comm, tree, decomp_domain, clon, start_model, tropomi_filename, cams_base_path, cams_params_file):
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

    # cams_times_sorted = sorted(cams_files_dict.keys())

    if comm.Get_rank() == 0:
        tropomi_ds = xr.open_dataset(tropomi_filename)
        raw_times_satellite = tropomi_ds['date'].values
        satellite_lons = tropomi_ds['lon'].values
        satellite_lats = tropomi_ds['lat'].values
        # ref_time = pd.to_datetime("2019-01-01 11:14:35.629000")
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
        cams_example_time = start.replace(hour=(start.hour // 6) * 6, minute=0, second=0, microsecond=0)
        example_file = cams_files_dict[cams_example_time]
        cams_ds = xr.open_dataset(example_file)
        cams_clon = np.asarray(cams_ds["clon"])
        cams_clat = np.asarray(cams_ds["clat"])
        cams_param_ds = xr.open_dataset(cams_params_file)
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

    (jc_loc_satellite, jb_loc_satellite, satellite_lons, satellite_lats, satellite_timestep, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_indices_sat, fracs_cams) = find_stations_satellite(satellite_lons, satellite_lats, obs_time_dts, tree, decomp_domain, clon, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree)
    # tc = []
    # tc_omv = [] # get the OMV-signal only in the profile

    # Preallocate arrays to avoid appending repeatedly
    # gridindices = []
    # obsindices = []
    # tcindices = []
    # ncount = 0

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
    done_lons_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_lats_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_times_sat = np.empty(N_satellite_points, dtype='datetime64[ns]')
    done_CH4_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_counter_sat = 0

    keys_to_do = ['lon', 'lat', 'timestep', 'jc_loc', 'jb_loc', 'pavg0', 'pw', 'ak', 'qa0', 'cams_index', 'frac_cams', 'hyam', 'hybm', 'hyai', 'hybi']
    values_to_do = [satellite_lons, satellite_lats, satellite_timestep, jc_loc_satellite, jb_loc_satellite, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_indices_sat, fracs_cams, hyam, hybm, hyai, hybi]
    local_data_to_do = {keys_to_do[i]:values_to_do[i] for i in range(len(keys_to_do))}

    keys_done = ['lon', 'lat', 'timestep', 'CH4', 'counter']
    values_done = [done_lons_sat, done_lats_sat, done_times_sat, done_CH4_sat, done_counter_sat]
    local_data_done = {keys_done[i]:values_done[i] for i in range(len(keys_done))}
    
    return local_data_to_do, local_data_done, cams_files_dict


def update_cams(datetime, cams_files_dict, cams_prev_data=None, cams_next_data=None):
    cams_prev_time = pd.to_datetime(datetime)
    cams_next_time = cams_prev_time + datetimelib.timedelta(hours=6)
    if cams_prev_data is not None:
        cams_prev_data.close()
    if cams_next_data is not None:
        cams_next_data.close()
    cams_prev_data = xr.open_dataset(cams_files_dict[cams_prev_time])
    cams_next_data = xr.open_dataset(cams_files_dict[cams_next_time])
    return cams_prev_data, cams_next_data


def tracking_CH4_satellite(datetime, comm, CH4_EMIS_np, CH4_BG_np, pres_ifc_np, pres_np, data_to_do, data_done, cams_prev_data, cams_next_data, cams_files_dict):
    if data_to_do['timestep'].size > 0: # Checks if there is still work to do this day

        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is greater or equal to the moment we want to measure. They are ready for measurement
        ready_mask = data_to_do['timestep'] <= model_time_np

        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready_sat = data_to_do['jc_loc'][ready_mask]
            jb_ready_sat = data_to_do['jb_loc'][ready_mask]
            frac_cams_ready = data_to_do['frac_cams'][ready_mask]
            frac_cams_ready = frac_cams_ready[:, np.newaxis]
            cams_indices_ready = data_to_do['cams_index'][ready_mask]

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
            hyam_new_axis, hybm_new_axis = data_to_do['hyam'][np.newaxis, :], data_to_do['hybm'][np.newaxis, :]
            hyai_new_axis, hybi_new_axis = data_to_do['hyai'][np.newaxis, :], data_to_do['hybi'][np.newaxis, :]
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

            pb_ret = data_to_do['pavg0'][ready_mask]
            # pb_ret = Tdata_cleaned.pressure_levels[nobs].values
            

            coef_matrix = np.array([get_int_coefs(pb_ret[i], pb_profile[i]) for i in range(pb_ret.shape[0])])

            # pwf = np.abs(np.diff(pb_ret) / np.ptp(pb_ret))
            pwf = data_to_do['pw'][ready_mask]
            averaging_kernel = data_to_do['ak'][ready_mask]
            important_stuff = data_to_do['qa0'][ready_mask]
            avpw = pwf * averaging_kernel
            prior_col = np.sum(pwf * important_stuff, axis=1)

            # print("RANK: ", rank, " coef_matrix shape: ", coef_matrix.shape, " , tracer_profile shape: ", tracer_profile.shape, file=sys.stderr)
            # profile_intrp = np.matmul(coef_matrix, tracer_profile)
            # profile_intrp = tracer_profile @ coef_matrix.T  # shape: (n_samples, 13)
            for i in range(len(tracer_profile)):
                if tracer_profile[i].shape[0] != pb_profile[i].shape[0] - 1:
                    print(f"RANK {comm.Get_rank()} — Mismatch at i={i}: tracer_profile = {tracer_profile[i].shape}, pb_profile = {pb_profile[i].shape}", file=sys.stderr)
            profile_intrp = np.matmul(coef_matrix, tracer_profile[..., np.newaxis])[..., 0]
            tc = prior_col + np.sum(avpw * (profile_intrp - important_stuff), axis=1)


            num_ready = np.sum(ready_mask) # Count how many points are ready
            done_counter = data_done['counter']

            # Add all of the done points to the done arrays
            data_done['lon'][done_counter:done_counter + num_ready] = data_to_do['lon'][ready_mask]
            data_done['lat'][done_counter:done_counter + num_ready] = data_to_do['lat'][ready_mask]
            data_done['timestep'][done_counter:done_counter + num_ready] = data_to_do['timestep'][ready_mask]
            data_done['CH4'][done_counter:done_counter + num_ready] = tc

            # Keep count of how many singlepoints are done
            data_done['counter'] += num_ready

            # Only keep the singlepoints that aren't done yet
            keep_mask = ~ready_mask

            keys_to_filter = [
                'lon', 'lat', 'timestep',
                'jc_loc', 'jb_loc', 'pavg0',
                'qa0', 'pw', 'ak', 'cams_index', 'frac_cams'
            ]

            for key in keys_to_filter:
                data_to_do[key] = data_to_do[key][keep_mask]
        
