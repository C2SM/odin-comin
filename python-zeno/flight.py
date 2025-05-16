"""
CH4 tracking plugin for the ICON Community Interface (ComIn)

@authors 04/2025 :: Zeno Hug, ICON Community Interface  <comin@icon-model.org>

SPDX-License-Identifier: BSD-3-Clause

Please see the file LICENSE in the root of the source tree for this code.
Where software is supplied by third parties, it is indicated in the
headers of the routines.
"""

import numpy as np
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


def find_stations_singlepoint(lons, lats, heights, are_abg, timesteps, tree, decomp_domain, clon, hhl, number_of_NN):
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

    # Loop through every station
    for lon, lat, height, above_ground, timestep in zip(lons, lats, heights, are_abg, timesteps):
        
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



def write_singlepoints(datetime, comm, done_flight_data):
    """Function to writeout the single timepoint data"""

    done_data_local = None
        #     keys_done = ['lon', 'lat', 'height', 'timestep', 'CH4', 'counter']
        # values_done = [done_lons, done_lats, done_heights, done_times, done_CH4, done_counter]
        # local_data_done = {[i]:values_done[i] for i in range(len(keys_done))}
    # Collect the local single point data, that we want to write out
    done_counter = done_flight_data['counter']
    if done_counter > 0:
        done_data_local = {
            "lon": done_flight_data['lon'][:done_counter],
            "lat": done_flight_data['lat'][:done_counter],
            "height": done_flight_data['height'][:done_counter],
            "timepoint": done_flight_data['timestep'][:done_counter],
            "CH4": done_flight_data['CH4'][:done_counter],
        }

    # Gather the local data to root 0, such that one process has all data that needs to be written out
    gathered_done_data = comm.gather(done_data_local, root=0)
    # The rank that has gathered the data will now write it out
    if comm.Get_rank() == 0:
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

    done_flight_data['counter'] = 0




def read_in_flight_data(datetime, comm, tree, decomp_domain, clon, hhl, number_of_NN):
    """Function to read in flight data and process the data to be ready for data tracking"""
    dt_str = pd.to_datetime(datetime).strftime("%Y%m%d")
    # Currently there is a manual list of days with flights. Will need to change later, but need to know how the naming structure of the flight data works
    # Predefine all variables
    singlepoint_lons = None
    singlepoint_lats = None
    singlepoint_heights = None
    singlepoint_is_abg = None
    singlepoint_timestep = None

    # Only 1 process reads it in as it could be problematic if a lot of processes try to read in the same file at the same time
    if comm.Get_rank() == 0:
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
            singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_is_abg, singlepoint_timestep, tree, decomp_domain, clon, hhl, number_of_NN)

    N_flight_points = singlepoint_lons.shape[0] # Amount of flight points in the local PE

    # Initialize all needed arrays as empty arrays of correct size
    done_lons = np.empty(N_flight_points, dtype=np.float64)
    done_lats = np.empty(N_flight_points, dtype=np.float64)
    done_heights = np.empty(N_flight_points, dtype=np.float64)
    done_times = np.empty(N_flight_points, dtype='datetime64[ns]')
    done_CH4 = np.empty(N_flight_points, dtype=np.float64)

    done_counter = 0 # counter of how many of the N_flight_points are already done (This day)


    keys_to_do = ['lon', 'lat', 'height', 'timestep', 'jc_loc', 'jb_loc', 'vertical_index1', 'vertical_index2', 'vertical_weight', 'horizontal_weight']
    values_to_do = [singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_timestep, jc_loc_singlepoint, jb_loc_singlepoint, vertical_indices_singlepoint1, vertical_indices_singlepoint2, vertical_weight_singlepoint, weights_singlepoint]
    local_data_to_do = {keys_to_do[i]:values_to_do[i] for i in range(len(keys_to_do))}

    keys_done = ['lon', 'lat', 'height', 'timestep', 'CH4', 'counter']
    values_done = [done_lons, done_lats, done_heights, done_times, done_CH4, done_counter]
    local_data_done = {keys_done[i]:values_done[i] for i in range(len(keys_done))}
    
    return local_data_to_do, local_data_done

def initialize_empty():
     """Helper function to just initialize empty"""
     return {'timestep': np.array([])}, {'counter': 0}

def tracking_CH4_flight(datetime, CH4_EMIS_np, CH4_BG_np, data_to_do, data_done):
     """The actual tracking of the CH4"""
     if data_to_do['timestep'].size > 0: # Checks if there is still work to do this day
        
        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is greater or equal to the moment we want to measure. They are ready for measurement
        ready_mask = data_to_do['timestep'] <= model_time_np

        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready = data_to_do['jc_loc'][ready_mask]
            jb_ready = data_to_do['jb_loc'][ready_mask]
            vi_ready1 = data_to_do['vertical_index1'][ready_mask]
            vi_ready2 = data_to_do['vertical_index2'][ready_mask]
            weights_vertical_ready = data_to_do['vertical_weight'][ready_mask]
            weights_ready = data_to_do['horizontal_weight'][ready_mask]
            done_counter = data_done['counter']

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
            data_done['lon'][done_counter:done_counter + num_ready] = data_to_do['lon'][ready_mask]
            data_done['lat'][done_counter:done_counter + num_ready] = data_to_do['lat'][ready_mask]
            data_done['height'][done_counter:done_counter + num_ready] = data_to_do['height'][ready_mask]
            data_done['timestep'][done_counter:done_counter + num_ready] = data_to_do['timestep'][ready_mask]
            data_done['CH4'][done_counter:done_counter + num_ready] = CH4_ready

            # Keep count of how many singlepoints are done
            data_done['counter'] += num_ready

            # Only keep the singlepoints that aren't done yet
            keep_mask = ~ready_mask

            keys_to_filter = [
                'lon', 'lat', 'height', 'timestep',
                'jc_loc', 'jb_loc', 'vertical_index1',
                'vertical_index2', 'vertical_weight', 'horizontal_weight'
            ]

            for key in keys_to_filter:
                data_to_do[key] = data_to_do[key][keep_mask]
    