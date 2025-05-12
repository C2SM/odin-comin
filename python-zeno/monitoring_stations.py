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
# import datetime as datetimelib
import os

def lonlat2xyz(lon, lat):
    """Short helper function for calculating xyz coordinates from longitues and latitudes"""
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

def find_stations_monitor(lons, lats, heights, are_abg, tree, decomp_domain, clon, hhl, number_of_NN):
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

    # Loop through every station
    for lon, lat, height, above_ground in zip(lons, lats, heights, are_abg):

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


def write_monitoring_stations(datetime, comm, data_monitoring, first_write_done_monitoring):
    """Function to writeout the stationary monitoring stations"""

    # Calculate averaged CH4
    avg_CH4_local = data_monitoring['CH4'] / data_monitoring['number_of_steps']
    avg_CH4_local = np.asarray(avg_CH4_local).ravel()

    # Gather everything on root 0
    gathered_avg_CH4 = comm.gather(avg_CH4_local, root=0)
    gathered_lons = comm.gather(data_monitoring['lon'], root=0)
    gathered_lats = comm.gather(data_monitoring['lat'], root=0)
    gathered_heights = comm.gather(data_monitoring['height'], root=0)

    # On the PE that has all the gathered data
    if comm.Get_rank() == 0:
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
    
    data_monitoring['number_of_steps'] = 0
    data_monitoring['CH4'][:] = 0
    return True

def read_in_monitoring_stations(datetime, comm, tree, decomp_domain, clon, hhl, number_of_NN):
    monitoring_lons = np.array([26.0, 23.0]) # predefine the monitoring stations. This could in future also be done via file inread
    monitoring_lats = np.array([46.0, 47.0])
    monitoring_heights = np.array([0.0, 0.0])
    monitoring_is_abg = np.array([True, True])
     # Find all of the monitoring stations in this local PE's domain and save all relevant data
    (jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring1, vertical_indices_monitoring2, vertical_weight_monitoring,  weights_monitoring, 
        monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg) = find_stations_monitor(
        monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg, tree, decomp_domain, clon, hhl, number_of_NN)

    current_CH4_monitoring = np.zeros(monitoring_lons.shape, dtype=np.float64) # Initialize the array for the CH4 monitoring to 0
    number_of_timesteps = 0
    keys = ['lon', 'lat', 'height', 'jc_loc', 'jb_loc', 'vertical_index1', 'vertical_index2', 'vertical_weight', 'horizontal_weight', 'CH4', 'number_of_steps']
    values = [monitoring_lons, monitoring_lats, monitoring_heights, jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring1, vertical_indices_monitoring2, vertical_weight_monitoring, weights_monitoring, current_CH4_monitoring, number_of_timesteps]
    data = {keys[i]:values[i] for i in range(len(keys))}

    return data

def tracking_CH4_monitoring(datetime, CH4_EMIS_np, CH4_BG_np, data):
    ## First we do the stationary monitoring
    # Fetch CH4 values in the correct indices, this fetches per monitoring station NUMBER_OF_NN points
    # Also, we want the CH4 Emissions in ppb. And the EMIS is not yet in ppb but just in parts per part. So we multiply by 1e9
    CH4_monitoring_all1 = (
        CH4_EMIS_np[data['jc_loc'], data['vertical_index1'], data['jb_loc'], 0, 0] * 1e9 +
        CH4_BG_np[data['jc_loc'], data['vertical_index1'], data['jb_loc'], 0, 0]
    )
    CH4_monitoring_all2 = (
        CH4_EMIS_np[data['jc_loc'], data['vertical_index2'], data['jb_loc'], 0, 0] * 1e9 +
        CH4_BG_np[data['jc_loc'], data['vertical_index2'], data['jb_loc'], 0, 0]
    )
    CH4_monitoring_all = CH4_monitoring_all1 + data['vertical_weight'] * (CH4_monitoring_all2 - CH4_monitoring_all1)
    # If we have any data we add the current contribution while also multiplying by the weights
    if data['horizontal_weight'].size > 0 and CH4_monitoring_all.size > 0:
        data['CH4'] += np.sum(data['horizontal_weight'] * CH4_monitoring_all, axis=1)

    data['number_of_steps'] += 1
        
