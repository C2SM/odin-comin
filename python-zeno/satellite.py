##
# @file satellite.py
#
# @brief Tracking any CH4 data to compare to satellite data
#
# @section description_satellite Description
# Tracking any CH4 data to compare to satellite data.
#
# @section libraries_satellite Libraries/Modules
# - numpy library
# - xarray library
# - netCDF4 library
#   - Access to Dataset
# - pandas library
#   - Access to pandas.to_datetime
# - os library
# - datetime library
# - scipy library
#   - Access to scipy.spatial.KDTree
# - get_int_coefs module (local)
#
# @section author_satellite Author(s)
# - Created by Zeno Hug on 05/23/2025.
#
# Copyright (c) 2025 Empa.  All rights reserved.

import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import KDTree
import datetime as datetimelib
from get_int_coefs import *
import os

## Constants:
Mda, MCH4 = 28.964, 16.04

def datetime_to_milliseconds_since_reference(arr, reference_str="2019-01-01T11:14:35.629"):
    """! Converts datetime 64 to milliseconds since a reference time.
    @param arr              An array of type datetime 64  to convert
    @param reference_str    Reference time in string format
    @return  The converted array
    """
    reference = np.datetime64(reference_str, 'ms')
    
    # Ensure array is datetime64[ms] for correct subtraction
    arr = np.asarray(arr).astype('datetime64[ms]')
    delta = arr - reference
    return delta.astype('timedelta64[ms]').astype('uint64')


def lonlat2xyz(lon, lat):
    """! Short helper function for calculating xyz coordinates from longitues and latitudes
    @param lon   An array or single value of longitudes to convert to xyz coordinates
    @param lat   An array or single value of latitudes to convert to xyz coordinates
    @return  converted xyz values as a tuple
    """
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)


def find_stations_satellite(lons, lats, timesteps, tree, decomp_domain, clon, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree):
    """! Find the satellite observation points on each PE in the own domain and return all of the relevant data needed for computation. All lists need to have the same length
    @param lons                 A list of longitudes of points to locate
    @param lats                 A list of latitudes of points to locate
    @param timesteps            A list of timesteps at which the measurements have been taken of the points needed to be located
    @param tree                 A tree with the cells in them that you can query
    @param decomp_domain        Array with information about which cells are in this PE's prognostic area
    @param clon                 Array with the cell longitudes
    @param pavg0_sat            A list of the pavg0 values of points to locate
    @param pw_sat               A list of the pw values of points to locate
    @param ak_sat               A list of the ak values of points to locate
    @param qa0_sat              A list of the qa0 values of points to locate
    @param cams_tree            A tree with the cells of the cams data in them that you can query
    @return  All of the relevant data of the points found in this PE's prognostic area: jb_loc, jc_loc, lon, lat, timesteps, pavg0, pw, ak, qa0, cams_indicesm fracs_cams
    """
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
        
        # Query the tree for the nearest cell
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = 1)

        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0]] == 0:
            jc_loc, jb_loc = np.unravel_index(ii[0], clon.shape) # Extract the indexes

            # Compute frac_cams and cams_index for later appending cams data to the model data
            cams_distances, cams_index = cams_tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = 1)
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


def write_satellite(comm, done_data, file_name_output):
    """!Function to write satellite data to output nc file using preallocated arrays with a counter.
    @param comm                 MPI communicator containing all working PE's
    @param data_done            Dictionary with the data that is done
    @param file_name_output     Filename of the output nc file. Expects the header of the nc file to be written already
    """
    done_data_local = None
    done_counter = done_data['counter']
    # Collect the local point data, that we want to write out
    if done_counter > 0:
        done_data_local = {
            "lon": done_data['lon'][:done_counter],
            "lat": done_data['lat'][:done_counter],
            "date": done_data['timestep'][:done_counter],
            "CH4": done_data['CH4'][:done_counter],
        }

    # Gather the local data to root 0, such that one process has all data that needs to be written out
    gathered_done_data = comm.gather(done_data_local, root=0)

    # The rank that has gathered the data will now write it out
    if comm.Get_rank() == 0:
        final_data = {
            "lon": [],
            "lat": [],
            "date": [],
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
            sort_indices = np.argsort(final_data["date"])
            for key in final_data:
                final_data[key] = final_data[key][sort_indices]
            ncfile = Dataset(file_name_output, 'a')
            obs_index = ncfile.variables['lon'].shape[0]
            new_points = final_data['lon'].shape[0]

            final_data['date'] = datetime_to_milliseconds_since_reference(final_data['date'])

            for var_name, data in final_data.items():
                    ncfile.variables[var_name][obs_index: obs_index + new_points] = data
            
            # Write out the data
            ncfile.close()
            
    done_data['counter'] = 0 # Reset the done counter

def read_in_satellite_data(comm, tree, decomp_domain, clon, start_model, end_model, tropomi_filename, cams_base_path, cams_params_file):
    """! Read in the satellite measurement points on each PE in the own domain and return all of the relevant data needed for computation as dictionaries
    @param comm                 MPI communicator containing all working PE's
    @param tree                 A tree with the cells in them that you can query
    @param decomp_domain        Array with information about which cells are in this PE's prognostic area
    @param clon                 Array with the cell longitudes
    @param tropomi_filename     Path to the tropomi nc file
    @param cams_base_path       Base path of the CAMS data
    @param cams_params_file     path to the CAMS parameter file
    @param start_model          datetime object depicting the start datetime of the experiment
    @param end_model            datetime object depicting the end datetime of the experiment
    @return  Tuple of dicts that symbolize the data to do, the data done and the cams files dict
    """
    cams_files_dict = {} # Dictionary for the cams file names

    # Create the dictionary for the cams file names
    dt = start_model.replace(hour=(start_model.hour // 6))
    while dt < end_model:
        dt_str = dt.strftime("%Y%m%d%H")
        fname = f"cams73_v22r2_ch4_conc_surface_inst_{dt_str}_lbc.nc"
        full_path = os.path.join(cams_base_path, fname)
        cams_files_dict[dt] = full_path
        dt += datetimelib.timedelta(hours=6)

    if comm.Get_rank() == 0:
        tropomi_ds = xr.open_dataset(tropomi_filename)
        # Extract all the needed values
        times_sat = tropomi_ds['date'].values
        satellite_lons = tropomi_ds['lon'].values
        satellite_lats = tropomi_ds['lat'].values
        pavg0_sat = tropomi_ds['pavg0'].values
        qa0_sat = tropomi_ds['qa0'].values
        ak_sat = tropomi_ds['ak'].values
        pw_sat = tropomi_ds['pw'].values
        obs_time_dts = pd.to_datetime(times_sat)
        
        # Only keep the points, where the time is between the starting and ending time
        valid_mask = (obs_time_dts >= start_model) & (obs_time_dts <= end_model)
        satellite_lons = satellite_lons[valid_mask]
        satellite_lats = satellite_lats[valid_mask]
        pavg0_sat = pavg0_sat[valid_mask]
        pw_sat = pw_sat[valid_mask]
        ak_sat = ak_sat[valid_mask]
        qa0_sat = qa0_sat[valid_mask]
        obs_time_dts = obs_time_dts[valid_mask]

        # Load one example CAMS file, for computing the correct CAMS indices
        cams_example_time = start_model.replace(hour=(start_model.hour // 6) * 6, minute=0, second=0, microsecond=0)
        example_file = cams_files_dict[cams_example_time]
        cams_ds = xr.open_dataset(example_file)
        cams_clon = np.asarray(cams_ds["clon"])
        cams_clat = np.asarray(cams_ds["clat"])

        # Load a CAMS parameter file with parameters needed later
        cams_param_ds = xr.open_dataset(cams_params_file)
        hyam = cams_param_ds["hyam"].values
        hybm = cams_param_ds["hybm"].values
        hyai = cams_param_ds["hyai"].values
        hybi = cams_param_ds["hybi"].values

        # Reverse CAMS hybrid coefficients so that they go from TOA to surface
        hyam = hyam[::-1]
        hybm = hybm[::-1]
        hyai = hyai[::-1]
        hybi = hybi[::-1]
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
    satellite_lons = comm.bcast(satellite_lons, root = 0)
    satellite_lats = comm.bcast(satellite_lats, root = 0)
    obs_time_dts = comm.bcast(obs_time_dts, root = 0)
    pavg0_sat = comm.bcast(pavg0_sat, root = 0)
    pw_sat = comm.bcast(pw_sat, root = 0)
    ak_sat = comm.bcast(ak_sat, root = 0)
    qa0_sat = comm.bcast(qa0_sat, root = 0)
    cams_clat = comm.bcast(cams_clat, root = 0)
    cams_clon = comm.bcast(cams_clon, root = 0)
    hyam = comm.bcast(hyam, root = 0)
    hybm = comm.bcast(hybm, root = 0)
    hyai = comm.bcast(hyai, root = 0)
    hybi = comm.bcast(hybi, root = 0)

    # Create the cams tree for searching later
    cams_xyz = np.c_[lonlat2xyz(cams_clon, cams_clat)]
    cams_tree = KDTree(cams_xyz)

    (jc_loc_satellite, jb_loc_satellite, satellite_lons, satellite_lats, satellite_timestep, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_indices_sat, fracs_cams) = find_stations_satellite(satellite_lons, satellite_lats, obs_time_dts, tree, decomp_domain, clon, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree)
   
    N_satellite_points = satellite_lons.shape[0] # Amount of satellite points in the local PE

    # Initialize all needed arrays as empty arrays of correct size
    done_lons_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_lats_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_times_sat = np.empty(N_satellite_points, dtype='datetime64[ns]')
    done_CH4_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_counter_sat = 0

    # Load all data into Dicts
    keys_to_do = ['lon', 'lat', 'timestep', 'jc_loc', 'jb_loc', 'pavg0', 'pw', 'ak', 'qa0', 'cams_index', 'frac_cams', 'hyam', 'hybm', 'hyai', 'hybi']
    values_to_do = [satellite_lons, satellite_lats, satellite_timestep, jc_loc_satellite, jb_loc_satellite, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_indices_sat, fracs_cams, hyam, hybm, hyai, hybi]
    local_data_to_do = {keys_to_do[i]:values_to_do[i] for i in range(len(keys_to_do))}

    keys_done = ['lon', 'lat', 'timestep', 'CH4', 'counter']
    values_done = [done_lons_sat, done_lats_sat, done_times_sat, done_CH4_sat, done_counter_sat]
    local_data_done = {keys_done[i]:values_done[i] for i in range(len(keys_done))}
    
    return local_data_to_do, local_data_done, cams_files_dict # Return all of the Data in Dicts


def update_cams(datetime, cams_files_dict, cams_prev_data=None, cams_next_data=None):
    """! Short function to update the current cams data, needs to be updated every 6 hours at 0, 6, 12 and 18
    @param datetime         Current datetime
    @param cams_file_dict   The dictionary with the CAMS file names
    @param cams_prev_data   prev data from before, will be closed
    @param cams_next_data   next data from before, will be closed
    @return Tuple of the updated CAMS data; Shape: prev_data, next_data
    """
    cams_prev_time = pd.to_datetime(datetime)
    cams_next_time = cams_prev_time + datetimelib.timedelta(hours=6)
    if cams_prev_data is not None:
        cams_prev_data.close()
    if cams_next_data is not None:
        cams_next_data.close()
    cams_prev_data = xr.open_dataset(cams_files_dict[cams_prev_time])
    cams_next_data = xr.open_dataset(cams_files_dict[cams_next_time])
    return cams_prev_data, cams_next_data


def tracking_CH4_satellite(datetime, CH4_EMIS_np, CH4_BG_np, pres_ifc_np, pres_np, data_to_do, data_done, cams_prev_data, cams_next_data):
    """! Track the CH4 of the satellite measurements. Move data that is done being measured to the data_done dictionary
    @param datetime             Current datetime (np.datetime object)
    @param data_to_do           The dictionary with the data that needs to be done
    @param data_done            The dictionary with the data that is done
    @param CH4_EMIS_np          Numpy array with the current CH4 EMIS
    @param CH4_BG_np            Numpy array with the current CH4 BG
    @param pres_ifc_np          Numpy array with the current pressures on the interfaces (so the half levels)
    @param pres_np              Numpy array with the current pressures
    @param cams_prev_data       Xarray dataset, with the current CAMS previous data
    @param cams_next_data       Xarray dataset, with the current CAMS next data
    """
    if data_to_do['timestep'].size > 0: # Checks if there is still work to do

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

            ## In general the following applies. All data is from TOA to surface. Up until the get coef function, which expects the data the other way around
            ## So there we turn everything around
            
            # Extract the pressure and CAMS data. For the CAMS data interpolate linearly between the 6 hour intervals
            pb_mod = (pres_ifc_np[jc_ready_sat, :, jb_ready_sat].squeeze()) / 1.e2
            pb_mod_mc =(pres_np[jc_ready_sat, :, jb_ready_sat].squeeze()) / 1.e2

            cams_indices_ready = cams_indices_ready.flatten()
            CAMS_obs_prev = cams_prev_data["CH4"].isel(time=0, ncells = cams_indices_ready).values[::-1].T
            CAMS_obs_next = cams_next_data["CH4"].isel(time=0, ncells = cams_indices_ready).values[::-1].T

            CAMS_aps_prev = cams_prev_data["ps"].isel(time=0, ncells = cams_indices_ready).values
            CAMS_aps_next = cams_next_data["ps"].isel(time=0, ncells = cams_indices_ready).values

            CAMS_aps_prev_reshaped = CAMS_aps_prev[:, np.newaxis]
            CAMS_aps_next_reshaped = CAMS_aps_next[:, np.newaxis]
            N_ready = CAMS_aps_prev.shape[0]  # number of ready satellite points

            hyam_new_axis = np.tile(data_to_do['hyam'], (N_ready, 1))
            hybm_new_axis = np.tile(data_to_do['hybm'], (N_ready, 1))
            hyai_new_axis = np.tile(data_to_do['hyai'], (N_ready, 1))
            hybi_new_axis = np.tile(data_to_do['hybi'], (N_ready, 1))
            # This is a formula I got from the CAMS data
            CAMS_p_prev =  (hyam_new_axis + hybm_new_axis * CAMS_aps_prev_reshaped) / 1.e2
            CAMS_p_next = (hyam_new_axis + hybm_new_axis * CAMS_aps_next_reshaped) / 1.e2
            CAMS_pressures = (1. - frac_cams_ready) * CAMS_p_prev + frac_cams_ready * CAMS_p_next

            CAMS_i_prev = (hyai_new_axis + hybi_new_axis * CAMS_aps_prev_reshaped) / 1.e2
            CAMS_i_next =  (hyai_new_axis + hybi_new_axis * CAMS_aps_next_reshaped) / 1.e2
            CAMS_interfaces = (1. - frac_cams_ready) * CAMS_i_prev + frac_cams_ready * CAMS_i_next
        
            # Extract the ICON CH4 profile, multiplying by 1e9 for the correct unit (ppb)
            ICON_profile = (
                (Mda / MCH4) * CH4_BG_np[jc_ready_sat, :, jb_ready_sat].squeeze() \
                + 1.e9 * (Mda / MCH4) * CH4_EMIS_np[jc_ready_sat, :, jb_ready_sat].squeeze()
            )
            ICON_profile = np.squeeze(ICON_profile)
            
            # Linearly interpolate the CAMS data to the current timestep, as CAMS data is only every 6 hours
            CAMS_obs = (1. - frac_cams_ready) * CAMS_obs_prev + frac_cams_ready * CAMS_obs_next

            ## The following code takes the ICON data and extends it by the CAMS data, as the ICON data does not go to very low pressures
            CAMS_profile = np.array([
                CAMS_obs[i, CAMS_pressures[i] < np.min(pb_mod_mc[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            pb_cams_mc = np.array([
                CAMS_pressures[i, CAMS_pressures[i] < np.min(pb_mod_mc[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            pb_cams = np.array([
                CAMS_interfaces[i, CAMS_interfaces[i] < np.min(pb_mod[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            # Was sometimes buggy, added this for safety
            ICON_profile = np.atleast_2d(ICON_profile)
            CAMS_profile = np.atleast_2d(CAMS_profile)
            pb_mod = np.atleast_2d(pb_mod)
            pb_cams = np.atleast_2d(pb_cams)
            pb_mod_mc = np.atleast_2d(pb_mod_mc)
            pb_cams_mc = np.atleast_2d(pb_cams_mc)
            tracer_profile = np.array([
                np.concatenate([CAMS_profile[i], ICON_profile[i]])
                for i in range(len(CAMS_profile))
            ], dtype=object)

            pb_profile = np.array([
                np.concatenate([pb_cams[i], pb_mod[i]])
                for i in range(len(pb_cams))
            ], dtype=object)

            pb_mc_profile = np.array([
                np.concatenate([pb_cams_mc[i], pb_mod_mc[i]])
                for i in range(len(pb_cams_mc))
            ], dtype=object)

            # Now as we get to the get coef funciton we need to turn around all of the data to: from surface to TOA
            # Also the computation afterwards is based on Michael Steiners computation
            pb_ret = data_to_do['pavg0'][ready_mask]
            coef_matrix = []
            pb_ret = pb_ret[:, ::-1]
            pb_profile = pb_profile[:, ::-1]
            tracer_profile = tracer_profile[:, ::-1]
            for i in range(pb_ret.shape[0]):
                coefs = get_int_coefs(pb_ret[i], pb_profile[i])

                coef_matrix.append(coefs)

            coef_matrix = np.array(coef_matrix)

            pwf = data_to_do['pw'][ready_mask]
            pwf = pwf[:, ::-1]
            averaging_kernel = data_to_do['ak'][ready_mask]
            averaging_kernel = averaging_kernel[:, ::-1]
            important_stuff = data_to_do['qa0'][ready_mask]
            important_stuff = important_stuff[:, ::-1]
            avpw = pwf * averaging_kernel
            prior_col = np.sum(pwf * important_stuff, axis=1)

            profile_intrp = np.matmul(coef_matrix, tracer_profile[..., np.newaxis])[..., 0]
            tc = prior_col + np.sum(avpw * (profile_intrp - important_stuff), axis=1)


            num_ready = np.sum(ready_mask) # Count how many points are ready
            done_counter = data_done['counter']

            # Add all of the done points to the done arrays
            data_done['lon'][done_counter:done_counter + num_ready] = data_to_do['lon'][ready_mask]
            data_done['lat'][done_counter:done_counter + num_ready] = data_to_do['lat'][ready_mask]
            data_done['timestep'][done_counter:done_counter + num_ready] = data_to_do['timestep'][ready_mask]
            data_done['CH4'][done_counter:done_counter + num_ready] = tc

            # Keep count of how many satellite points are done
            data_done['counter'] += num_ready

            # Only keep the satellite points that aren't done yet
            keep_mask = ~ready_mask

            keys_to_filter = [
                'lon', 'lat', 'timestep',
                'jc_loc', 'jb_loc', 'pavg0',
                'qa0', 'pw', 'ak', 'cams_index', 'frac_cams'
            ]

            for key in keys_to_filter:
                data_to_do[key] = data_to_do[key][keep_mask]
        
