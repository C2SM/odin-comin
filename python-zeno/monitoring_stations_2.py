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
# from scipy.spatial import KDTree
import datetime as datetimelib
import os

def lonlat2xyz(lon, lat):
    """Short helper function for calculating xyz coordinates from longitues and latitudes"""
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

def find_stations_monitor2(lons, lats, heights_abs, are_abg, tree, decomp_domain, clon, hhl, number_of_NN, ids, timesteps, heights_abg):
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
    ids_local = []
    timesteps_local = []

    # Loop through every station
    for lon, lat, height_abs, above_ground, id, timestep, height_abg in zip(lons, lats, heights_abs, are_abg, ids, timesteps, heights_abg):

        # Query the tree for the NUMBER_OF_NN nearest cells
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = number_of_NN)

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
            height = height_abs

            # Now, we want to compute the correct vertical index for each of the NUMBER_OF_NN cells. 
            for jc, jb in zip(jc_loc, jb_loc):
                local_hhl = hhl[jc, :, jb].squeeze() # This is the vertical column of half height levels
                # As the hhl are half height levels we want to get the height levels of the cells. This is done by always taking the middle between the hhls
                h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])
                
                # As the height in the model is in height above sea, we want to add the lowest level hhl (which is the ground level) if the height is measured above ground
                # Also depending on if we catgeorized it as a mountain or lowland. For lowland we take the abg measurement, and as we need the abs level, we add ground level
                # For the mountains we take the height abs, but we want to add 50% of the difference between abg and abs
                height_above_sea = 0
                if above_ground:
                    height_above_sea = height_abg + local_hhl[-1]
                else:
                    height_above_sea = height_abs
                    height_above_sea_abg = height_abg + local_hhl[-1]
                    closest_approx_abg = np.min(np.abs(h_mid - height_above_sea_abg ))
                    height_above_sea = height_above_sea + 0.5 * (closest_approx_abg - height_above_sea)


                closest_index = int(np.argmin(np.abs(h_mid - height_above_sea)))
                
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
            while len(jc_row) < number_of_NN:
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
            ids_local.append(id)
            timesteps_local.append(timestep)


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
            np.array(ids_local, dtype = str), 
            np.array(timesteps_local))


def write_monitoring_stations2(datetime, comm, data_done, first_write_done_monitoring):
    """Function to writeout the stationary monitoring stations"""

    # Gather everything on root 0
    gathered_avg_CH4 = comm.gather(data_done['CH4'], root=0)
    gathered_lons = comm.gather(data_done['lon'], root=0)
    gathered_lats = comm.gather(data_done['lat'], root=0)
    gathered_heights = comm.gather(data_done['height'], root=0)
    gathered_times = comm.gather(data_done['timestep'], root=0)
    gathered_ids = comm.gather(data_done['id'], root=0)

    # On the PE that has all the gathered data
    if comm.Get_rank() == 0:
        # Flatten the data
        avg_CH4_flat = np.concatenate(gathered_avg_CH4)
        lons_flat = np.concatenate(gathered_lons)
        lats_flat = np.concatenate(gathered_lats)
        heights_flat = np.concatenate(gathered_heights)
        ids_flat = np.concatenate(gathered_ids)
        timesteps_flat = np.concatenate(gathered_times)

        station_ids = ids_flat

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
                "time": ("time", timesteps_flat),
            },
        )

        # Metadata, set units etc.
        ds["avg_CH4"].attrs["units"] = "ppb"
        ds["avg_CH4"].attrs["long_name"] = "Average CH4 concentration over last hour"
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


        output_file = "tracked_ch4_2.nc" # filename of the nc file

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
    
    data_done['counter'] = 0
    return first_write_done_monitoring

def read_in_monitoring_stations2(datetime, comm, tree, decomp_domain, clon, hhl, number_of_NN, final_stationslist_filename, path_to_csvs):
    if comm.Get_rank() == 0:
        # Read the CSV with the required metadata
        meta_info = pd.read_csv(final_stationslist_filename, sep=',', comment='#')

        # Drop completely empty rows
        meta_info.dropna(how='all', inplace=True)

        files_list = meta_info['obsfile'] # List of files of which we want to extract the times and lats/Lons
        sampling_height = meta_info['sampling_height'] # Meta info about the height. Is either 1 or 2

        monitoring_lons = [] # predefine the lists
        monitoring_lats = []
        monitoring_heights = []
        monitoring_is_abg = []
        monitoring_ids = []
        monitoring_timesteps = []
        monitoring_heights_abg = []
        # Manually define the column names, as they arent seperated by commas, like the actual data, so we can't read them in correctly
        column_names = ["siteid", "year", "month", "day", "hour", "minute", "second", "value", "value_unc", "nvalue", "latitude", "longitude", "altitude", "elevation", "intake_height", "QCflag", "scale"]
        for file_name, meta_info_height in zip(files_list, sampling_height):
            full_path = os.path.join(path_to_csvs, file_name)
            this_df = pd.read_csv(full_path, sep = ',', header = 9, skipinitialspace = True, names = column_names)
            this_lats = this_df['latitude'].to_list()
            this_lons = this_df['longitude'].to_list()
            this_id = this_df['siteid'].to_list()
            this_df['timestamp'] = pd.to_datetime(this_df[['year', 'month', 'day', 'hour', 'minute', 'second']])
            this_timestep = this_df['timestamp'].to_list()
            this_height_abg = this_df['intake_height'].to_list()
            this_height = this_df['altitude'].to_list()
            if meta_info_height == 1:
                this_is_abg = np.ones(len(this_height), dtype = bool)
            elif meta_info_height == 2:
                this_is_abg = np.zeros(len(this_height), dtype = bool)
            # Only keep the points, where the time is later (or equal) than now
            valid_mask = this_timestep >= pd.to_datetime(datetime)
            this_lats = this_lats[valid_mask]
            this_lons = this_lons[valid_mask]
            this_id = this_id[valid_mask]
            this_height_abg = this_height_abg[valid_mask]
            this_height = this_height[valid_mask]
            this_is_abg = this_is_abg[valid_mask]
            this_timestep = this_timestep[valid_mask]
            
            monitoring_lons.extend(this_lons)
            monitoring_lats.extend(this_lats)
            monitoring_ids.extend(this_id)
            monitoring_is_abg.extend(this_is_abg)
            monitoring_heights.extend(this_height)
            monitoring_timesteps.extend(this_timestep)
            monitoring_heights_abg.extend(this_height_abg)

        
        monitoring_lons = np.array(monitoring_lons)
        monitoring_lats = np.array(monitoring_lats)
        monitoring_ids = np.array(monitoring_ids)
        monitoring_is_abg = np.array(monitoring_is_abg)
        monitoring_heights = np.array(monitoring_heights)
        monitoring_timesteps = np.array(monitoring_timesteps)
        monitoring_heights_abg = np.array(monitoring_heights_abg)
    else:
        monitoring_lons = None
        monitoring_lats = None
        monitoring_ids = None
        monitoring_is_abg = None
        monitoring_heights = None
        monitoring_timesteps = None
        monitoring_heights_abg = None
    
    # Broadcast the data to all processes, from root 0
    monitoring_lons = comm.bcast(monitoring_lons, root=0)
    monitoring_lats = comm.bcast(monitoring_lats, root=0)
    monitoring_ids = comm.bcast(monitoring_ids, root=0)
    monitoring_is_abg = comm.bcast(monitoring_is_abg, root=0)
    monitoring_heights = comm.bcast(monitoring_heights, root=0)
    monitoring_timesteps = comm.bcast(monitoring_timesteps, root=0)
    monitoring_heights_abg = comm.bcast(monitoring_heights_abg, root=0)
  
     # Find all of the monitoring stations in this local PE's domain and save all relevant data
    (jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring1, vertical_indices_monitoring2, vertical_weight_monitoring,  weights_monitoring, 
        monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg, monitoring_ids, monitoring_timesteps) = find_stations_monitor2(
        monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg, tree, decomp_domain, clon, hhl, number_of_NN, monitoring_ids, monitoring_timesteps, monitoring_heights_abg)

    current_CH4_monitoring = np.zeros(monitoring_lons.shape, dtype=np.float64)
    N_points = monitoring_lons.shape[0]

    # Initialize all needed arrays as empty arrays of correct size and type
    done_lons = np.empty(N_points, dtype=np.float64)
    done_lats = np.empty(N_points, dtype=np.float64)
    done_heights = np.empty(N_points, dtype=np.float64)
    done_times = np.empty(N_points, dtype='datetime64[ns]')
    done_CH4 = np.empty(N_points, dtype=np.float64)
    done_ids = np.empty(N_points, dtype=str)

    done_counter = 0 # counter of how many of the points are already done (since last writeout)

    ## Create Dicts with all of the data needed
    number_of_timesteps = np.zeros(N_points, dtype=np.int32)
    keys = ['lon', 'lat', 'height', 'jc_loc', 'jb_loc', 'vertical_index1', 'vertical_index2', 'vertical_weight', 'horizontal_weight', 'CH4', 'number_of_steps', 'id', 'timestep']
    values = [monitoring_lons, monitoring_lats, monitoring_heights, jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring1, vertical_indices_monitoring2, vertical_weight_monitoring, weights_monitoring, current_CH4_monitoring, number_of_timesteps, monitoring_ids, monitoring_timesteps]
    data = {keys[i]:values[i] for i in range(len(keys))}

    keys_done = ['lon', 'lat', 'height', 'timestep', 'CH4', 'counter', 'id']
    values_done = [done_lons, done_lats, done_heights, done_times, done_CH4, done_counter, done_ids]
    data_done = {keys_done[i]:values_done[i] for i in range(len(keys_done))}

    return data, data_done # Return the dicts with the data

def tracking_CH4_monitoring2(datetime, CH4_EMIS_np, CH4_BG_np, data_to_do, data_done):

    if data_to_do['timestep'].size > 0: # Checks if there is still work to do this day
        
        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is in the hour before the output of the measurement. They are ready for measurement
        measuring_mask = (((data_to_do['timestep'] - datetimelib.timedelta(hours=1)) <= model_time_np) & (data_to_do['timestep'] > model_time_np))

        if np.any(measuring_mask):
            # Filter arrays for ready stations
            jc_ready = data_to_do['jc_loc'][measuring_mask]
            jb_ready = data_to_do['jb_loc'][measuring_mask]
            vi_ready1 = data_to_do['vertical_index1'][measuring_mask]
            vi_ready2 = data_to_do['vertical_index2'][measuring_mask]
            weights_vertical_ready = data_to_do['vertical_weight'][measuring_mask]
            weights_ready = data_to_do['horizontal_weight'][measuring_mask]
            


            # we want the CH4 Emissions in ppb. And the EMIS is not yet in ppb but just in parts per part. So we multiply by 1e9
            CH4_monitoring_all1 = (
                CH4_EMIS_np[jc_ready, vi_ready1, jb_ready, 0, 0] * 1e9 +
                CH4_BG_np[jc_ready, vi_ready1, jb_ready, 0, 0]
            )
            CH4_monitoring_all2 = (
                CH4_EMIS_np[jc_ready, vi_ready2, jb_ready, 0, 0] * 1e9 +
                CH4_BG_np[jc_ready, vi_ready2, jb_ready, 0, 0]
            )
            CH4_monitoring_all = CH4_monitoring_all1 + weights_vertical_ready * (CH4_monitoring_all2 - CH4_monitoring_all1)
            # If we have any data we add the current contribution while also multiplying by the weights
            if weights_ready.size > 0 and CH4_monitoring_all.size > 0:
                data_to_do['CH4'] += np.sum(weights_ready * CH4_monitoring_all, axis=1)
                data_to_do['number_of_steps'][measuring_mask] += 1

        done_mask = data_to_do['timestep'] <= model_time_np # This data is done being monitored and can be output
        done_counter = data_done['counter']
        num_ready = np.sum(done_mask) # Count how many points are done
        if num_ready > 0:
        # Add all of the done points to the done arrays
            data_done['lon'][done_counter:done_counter + num_ready] = data_to_do['lon'][done_mask]
            data_done['lat'][done_counter:done_counter + num_ready] = data_to_do['lat'][done_mask]
            data_done['height'][done_counter:done_counter + num_ready] = data_to_do['height'][done_mask]
            data_done['timestep'][done_counter:done_counter + num_ready] = data_to_do['timestep'][done_mask]
            # zero_mask = (data_to_do['number_of_steps'][done_mask] == 0)
            # data_to_do['number_of_steps'][done_mask][zero_mask][:] = 1 # This code block would be to explictily not be able to divide by 0. I can also leave it and just get NaN
            # Current behaviour is that if there are no steps made, I divide by 0 which results in NaN, which is actually great as I then in post processing know for which points I dont have data
            data_done['CH4'][done_counter:done_counter + num_ready] = data_to_do['CH4'][done_mask] / data_to_do['number_of_steps'][done_mask]
            data_done['id'][done_counter:done_counter + num_ready] = data_to_do['id'][done_mask]

            # Keep count of how many points are done
            data_done['counter'] += num_ready

            # Only keep the points that aren't done yet
            keep_mask = ~done_mask
            keys_to_filter = [
                'lon', 'lat', 'height', 'timestep',
                'jc_loc', 'jb_loc', 'vertical_index1',
                'vertical_index2', 'vertical_weight', 'horizontal_weight', 'CH4', 'id', 'number_of_steps'
            ]
            # Filter the entrys in the dictionary and only keep the points that aren't done yet
            for key in keys_to_filter:
                data_to_do[key] = data_to_do[key][keep_mask]
        
