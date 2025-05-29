##
# @file monitoring_stations_final.py
#
# @brief Tracking any variables averaging over starting and ending times
#
# @section description_monitoring_stations_final Description
# Tracking any variables averaging over starting and ending times.
#
# @section libraries_monitoring_stations_final Libraries/Modules
# - numpy library
# - xarray library
# - netCDF4 library
#   - Access to Dataset
# - pandas library
#   - Access to pandas.to_datetime
# - sys library
#
# @section author_monitoring_stations_final Author(s)
# - Created by Zeno Hug on 05/23/2025.
#
# Copyright (c) 2025 Empa.  All rights reserved.


# Imports
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import pandas as pd
import sys

# Functions
def datetime64_to_days_since_1970(arr):
    """! Converts datetime 64 to days since 1970
    @param arr  An array or single value of type datetime 64  to convert
    @return  The converted array/single value 
    """
    epoch = np.datetime64("1970-01-01T00:00:00", 'ns')
    return (arr - epoch) / np.timedelta64(1, 'D')

def lonlat2xyz(lon, lat):
    """! Short helper function for calculating xyz coordinates from longitues and latitudes
    @param lon   An array or single value of longitudes to convert to xyz coordinates
    @param lat   An array or single value of latitudes to convert to xyz coordinates
    @return  converted xyz values as a tuple
    """
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

def find_points(lons, lats, sampling_heights, sampling_elevations, sampling_strategies, tree, decomp_domain, clon, hhl, number_of_NN, ids, timesteps_begin, timesteps_end, comm):
    """! Find the local stationary monitoring stations on each PE in the own domain and return all of the relevant data needed for computation. All lists need to have the same length
    @param lons                 A list of longitudes of points to locate
    @param lats                 A list of latitudes of points to locate
    @param sampling_heights     A list of sampling heights, meaning height above ground in meters
    @param sampling_elevations  A list of sampling elevations, meaning the height of the ground, in meters above sea level
    @param sampling_strategies  A list of sampling strategy flags, 1 = lowland, 2 = mountain, 3 = instantaneous measurement in lowland, 4 = averaged measurement in mountain
    @param tree                 A tree with the cells in them that you can query
    @param decomp_domain        Array with information about which cells are in this PE's prognostic area
    @param clon                 Array with the cell longitudes
    @param hhl                  Array with the height of the half levels
    @param number_of_NN         Number of nearest cells over which should be interpolated
    @param ids                  A list of ids (strings) of the points to locate
    @param timesteps_begin      A list of timepoints at which the measuring should start of the points to locate
    @param timesteps_end        A list of timepoints at which the measuring should end of the points to locate
    @return  All of the relevant data of the points found in this PE's prognostic area: jb_loc, jc_loc, vertical index 1 for vertical interpolation, vertical index 2 for vertical interpolation, weight for vertical interpolation, weights for horizontal interpolation, and all of the input lists with only the data stored of this PE's points
    """

    jc_locs = [] 
    jb_locs = []
    vertical_indices1 = []
    vertical_indices2 = []
    weights_vertical_all = []
    weights_all = []
    lons_local = []
    lats_local = []
    sampling_heights_local = []
    sampling_elevations_local = []
    sampling_strategy_local = []
    ids_local = []
    timesteps_local_begin = []
    timesteps_local_end = []

    # Loop through every station
    for lon, lat, sampling_height, sampling_elevation, sampling_strategy, id, timestep_begin, timestep_end in zip(lons, lats, sampling_heights, sampling_elevations, sampling_strategies, ids, timesteps_begin, timesteps_end):

        # Query the tree for the NUMBER_OF_NN nearest cells
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = number_of_NN)

        # jc_loc_1, jb_loc_1 = np.unravel_index(ii[0], clon.shape)
        # print("RANK: ", comm.Get_rank(), " JC LOC: ", jc_loc_1, " JB LOC: ", jb_loc_1, file=sys.stderr)

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

            for jc, jb in zip(jc_loc, jb_loc):
                local_hhl = hhl[jc, :, jb].squeeze() # This is the vertical column of half height levels
                # As the hhl are half height levels we want to get the height levels of the cells. This is done by always taking the middle between the hhls
                h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])
                
                # As the height in the model is in height above sea, we want to add the lowest level hhl (which is the ground level) if the height is measured above ground
                # Also depending on if we catgeorized it as a mountain or lowland. For lowland we take the abg measurement, and as we need the abs level, we add ground level
                # For the mountains we take the height abs, but we want to add 50% of the difference between abg and abs
                height_above_sea = 0
                if sampling_strategy == 1 or sampling_strategy == 3:
                    height_above_sea = sampling_height + local_hhl[-1]
                elif sampling_strategy == 2 or sampling_strategy == 4:
                    height_above_sea = sampling_height + sampling_elevation
                    model_ground_elevation = local_hhl[-1]
                    model_topo_elev = sampling_elevation + model_ground_elevation
                    topo_error = height_above_sea - model_topo_elev
                    height_above_sea = height_above_sea - 0.5 * topo_error



                closest_index = int(np.argmin(np.abs(h_mid - height_above_sea)))
                
                actual_height_closest = h_mid[closest_index]
                second_index = closest_index
                # Second index is for height interpolation. depending on where the closest height is, compute the second index, also taking into consideration boundaries
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
            if np.any(dd_local == 0):
                print('The longitude/latitude coincides identically with an ICON cell, which is an issue for the inverse distance weighting.', file=sys.stderr)
                print('I will slightly modify this value to avoid errors.', file=sys.stderr)
                dd_local[dd_local == 0] = 1e-12

            # Here we compute the weights for interpolating. The weights are normalized to sum up to 1 and they are proportional to the inverse of the distances
            weights = 1.0 / dd_local
            weights = weights / np.sum(weights)
            weight_row = weights.tolist()

            # If fewer than NUMBER_OF_NN neighbors, pad with 0. This should not happen, as long as the number of neighbors is not set too high but just to be safe
            # As the weight is set to 0 it does not affect the result
            while len(jc_row) < number_of_NN:
                jc_row.append(0)
                jb_row.append(0)
                vertical_row1.append(0)
                vertical_row2.append(0)
                weight_row_vertical.append(0.0)
                weight_row.append(0.0)

            # Append all data, for the cells that were found in this PE's domain
            jc_locs.append(jc_row)
            jb_locs.append(jb_row)
            weights_all.append(weight_row)
            lons_local.append(lon)
            lats_local.append(lat)
            sampling_elevations_local.append(sampling_elevation)
            sampling_strategy_local.append(sampling_strategy)
            sampling_heights_local.append(sampling_height)
            vertical_indices1.append(vertical_row1)
            vertical_indices2.append(vertical_row2)
            weights_vertical_all.append(weight_row_vertical)
            ids_local.append(id)
            timesteps_local_begin.append(timestep_begin)
            timesteps_local_end.append(timestep_end)


    # Return all data as numpy arrays
    return (np.array(jc_locs, dtype = np.int32),
            np.array(jb_locs, dtype = np.int32),
            np.array(vertical_indices1, dtype = np.int32),
            np.array(vertical_indices2, dtype = np.int32),
            np.array(weights_vertical_all, dtype = np.float64),
            np.array(weights_all, dtype = np.float64),
            np.array(lons_local, dtype = np.float64),
            np.array(lats_local, dtype = np.float64),
            np.array(sampling_elevations_local, dtype = np.float64),
            np.array(sampling_heights_local, dtype = np.float64),
            np.array(sampling_strategy_local, dtype = np.int32),
            np.array(ids_local, dtype = 'U20'), 
            np.array(timesteps_local_begin), 
            np.array(timesteps_local_end))


def write_points(comm, data_done, dict_vars, file_name_output):
    """!Function to write stationary monitoring data to output nc file using preallocated arrays with a counter.
    @param comm                 MPI communicator containing all working PE's
    @param data_done            Dictionary with the data that is done
    @param dict_vars            Dictionary with the variable info, user defined
    @param file_name_output     Filename of the output nc file. Expects the header of the nc file to be written already
    """
    done_counter = data_done['counter']
    done_data_local = None
    if done_counter > 0:

        done_data_local = {
            "site_name": data_done['id'][:done_counter],
            "longitude": data_done['lon'][:done_counter],
            "latitude": data_done['lat'][:done_counter],
            "elevation": data_done['elevation'][:done_counter],
            "sampling_height": data_done['sampling_height'][:done_counter],
            "sampling_strategy": data_done['sampling_strategy'][:done_counter],
            "stime": data_done['stime'][:done_counter],
            "etime": data_done['etime'][:done_counter],
        }

        for variable in dict_vars:
            done_data_local[variable] = data_done[variable][:done_counter]

    gathered_done_data = comm.gather(done_data_local, root=0)

    if comm.Get_rank() == 0:
        final_data = {
            "site_name": [],
            "longitude": [],
            "latitude": [],
            "elevation": [],
            "sampling_height": [],
            "sampling_strategy": [],
            "stime": [],
            "etime": [],
        }
        for variable in dict_vars:
            final_data[variable] = []

        for d in gathered_done_data:
            if d is not None:
                for key in final_data:
                    final_data[key].append(d[key])

        if any(len(lst) > 0 for lst in final_data.values()):
            for key in final_data:
                final_data[key] = np.concatenate(final_data[key])
            sort_indices = np.argsort(final_data["etime"])
            for key in final_data:
                final_data[key] = final_data[key][sort_indices]
            ncfile = Dataset(file_name_output, 'a')
            obs_index = ncfile.variables['longitude'].shape[0]
            new_points = final_data['longitude'].shape[0]
            final_data['site_name'] = np.array([list(name.ljust(20)[:20]) for name in final_data['site_name']],dtype='S1')
            final_data['stime'] = datetime64_to_days_since_1970(final_data['stime'])
            final_data['etime'] = datetime64_to_days_since_1970(final_data['etime'])

            for var_name, data in final_data.items():
                if var_name != 'site_name':
                    ncfile.variables[var_name][obs_index: obs_index + new_points] = data
                else:
                    ncfile.variables[var_name][obs_index: obs_index + new_points, :] = data
            
            ncfile.close()
            

    data_done['counter'] = 0


def write_header_points(comm, file_name, dict_vars):
    if(comm.Get_rank() == 0):
        ncfile = Dataset(file_name, 'w', format='NETCDF4')

        # Define dimensions
        ncfile.createDimension('obs', None)  # unlimited
        ncfile.createDimension('nchar', 20)

        # Define variables
        stime = ncfile.createVariable('stime', 'f8', ('obs',))
        stime.units = "days since 1970-01-01 00:00:00"
        stime.long_name = "start time of observation interval; UTC"
        stime.calendar = "proleptic_gregorian"

        etime = ncfile.createVariable('etime', 'f8', ('obs',))
        etime.units = "days since 1970-01-01 00:00:00"
        etime.long_name = "end time of observation interval; UTC"
        etime.calendar = "proleptic_gregorian"

        longitude = ncfile.createVariable('longitude', 'f4', ('obs',), fill_value=1.0e+20)
        longitude.units = "degrees_east"
        longitude.standard_name = "longitude"

        latitude = ncfile.createVariable('latitude', 'f4', ('obs',), fill_value=1.0e+20)
        latitude.units = "degrees_north"
        latitude.standard_name = "latitude"

        elevation = ncfile.createVariable('elevation', 'f4', ('obs',), fill_value=1.0e+20)
        elevation.units = "m"
        elevation.long_name = "surface elevation above sea level"

        sampling_height = ncfile.createVariable('sampling_height', 'f4', ('obs',), fill_value=1.0e+20)
        sampling_height.units = "m"
        sampling_height.long_name = "sampling height above surface"

        sampling_strategy = ncfile.createVariable('sampling_strategy', 'f4', ('obs',))
        sampling_strategy.units = "1"
        sampling_strategy.long_name = "sampling strategy flag"
        sampling_strategy.comment = "1=low ; 2=mountain ; 3=flight"

        site_name = ncfile.createVariable('site_name', 'S1', ('obs', 'nchar'))
        site_name.long_name = "station name or ID"

        # Global attributes
        ncfile.Conventions = "CF-1.8"
        ncfile.title = "Station input file for ICON ComIn interface XYZ"
        ncfile.institution = "Empa"
        ncfile.source = "ICON ComIn interface XYZ"
        ncfile.version = "1.0"
        ncfile.author = "Zeno Hug"
        ncfile.transport_model = "ICON"
        ncfile.transport_model_version = ""
        ncfile.experiment = ""
        ncfile.project = ""
        ncfile.references = ""
        ncfile.comment = ""
        ncfile.license = "CC-BY-4.0"
        ncfile.history = ""
        ncfile.close()
        for variable, parameters in dict_vars.items():
            ncfile = Dataset(file_name, 'a')
            temp_var = ncfile.createVariable(variable, 'f8', ('obs',))
            temp_var.units = parameters['unit']
            temp_var.long_name = parameters['long_name']
            ncfile.close()


def read_in_points(comm, tree, decomp_domain, clon, hhl, number_of_NN, path_to_file, start_model, end_model, data_vars):
    """! Read in the local stationary monitoring stations on each PE in the own domain and return all of the relevant data needed for computation as dictionaries
    @param comm                 MPI communicator containing all working PE's
    @param tree                 A tree with the cells in them that you can query
    @param decomp_domain        Array with information about which cells are in this PE's prognostic area
    @param clon                 Array with the cell longitudes
    @param hhl                  Array with the height of the half levels
    @param number_of_NN         Number of nearest cells over which should be interpolated
    @param path_to_file         The path of the input file
    @param start_model          datetime object depicting the start datetime of the experiment
    @param end_model            datetime object depicting the end datetime of the experiment
    @param data_vars            The data dictionary with the data from the simulation
    @return  Tuple of dicts that symbolize the data to do and the data done
    """
    if comm.Get_rank() == 0:
        input_ds = xr.open_dataset(path_to_file)
        # Extract all the needed values
        lons = input_ds['longitude'].values
        lats = input_ds['latitude'].values
        stimes = input_ds['stime'].values
        etimes = input_ds['etime'].values
        elevations = input_ds['elevation'].values
        sampling_heights = input_ds['sampling_height'].values
        sampling_strategies = input_ds['sampling_strategy'].values
        site_names = input_ds['site_name'].values
        stimes = pd.to_datetime(stimes)
        etimes = pd.to_datetime(etimes)
        
        # Only keep the points, where the time is between model start and model end
        valid_mask = (stimes >= start_model) & (etimes <= end_model)
        lons = lons[valid_mask]
        lats = lats[valid_mask]
        elevations = elevations[valid_mask]
        sampling_heights = sampling_heights[valid_mask]
        sampling_strategies = sampling_strategies[valid_mask]
        site_names = site_names[valid_mask]
        stimes = stimes[valid_mask]
        etimes = etimes[valid_mask]
    else:
        lons = None
        lats = None
        elevations = None
        sampling_heights = None
        sampling_strategies = None
        site_names = None
        stimes = None
        etimes = None

    
    # Broadcast the data to all processes, from root 0
    lons = comm.bcast(lons, root = 0)
    lats = comm.bcast(lats, root = 0)
    elevations = comm.bcast(elevations, root = 0)
    sampling_heights = comm.bcast(sampling_heights, root = 0)
    sampling_strategies = comm.bcast(sampling_strategies, root = 0)
    site_names = comm.bcast(site_names, root = 0)
    stimes = comm.bcast(stimes, root = 0)
    etimes = comm.bcast(etimes, root = 0)

     # Find all of the monitoring stations in this local PE's domain and save all relevant data
    (jc_loc, jb_loc, vertical_indices_nearest, vertical_indices_second, vertical_weights,  horizontal_weights, 
        lons, lats, elevations, sampling_heights, sampling_strategies, ids, stimes, etimes) = find_points(
        lons, lats, sampling_heights, elevations, sampling_strategies, tree, decomp_domain, clon, hhl, number_of_NN, site_names, stimes, etimes, comm)

    N_points = lons.shape[0]

    # Initialize all needed arrays as empty arrays of correct size and type
    done_lons = np.empty(N_points, dtype=np.float64)
    done_lats = np.empty(N_points, dtype=np.float64)
    done_elevations = np.empty(N_points, dtype=np.float64)
    done_sampling_heights = np.empty(N_points, dtype=np.float64)
    done_sampling_strategies = np.empty(N_points, dtype=np.int32)
    done_site_names = np.empty(N_points, dtype='U20')
    done_stimes = np.empty(N_points, dtype='datetime64[ns]')
    done_etimes = np.empty(N_points, dtype='datetime64[ns]')
    

    done_counter = 0 # counter of how many of the points are already done (since last writeout)

    # Create Dicts with all of the data needed
    number_of_timesteps = np.zeros(N_points, dtype=np.int32)
    keys = ['lon', 'lat','elevation', 'sampling_height', 'sampling_strategy', 'jc_loc', 'jb_loc', 'vertical_index1', 'vertical_index2', 'vertical_weight', 'horizontal_weight', 'number_of_steps', 'id', 'stime', 'etime']
    values = [lons, lats, elevations, sampling_heights, sampling_strategies, jc_loc, jb_loc, vertical_indices_nearest, vertical_indices_second, vertical_weights, horizontal_weights, number_of_timesteps, ids, stimes, etimes]
    data = {keys[i]:values[i] for i in range(len(keys))}

    keys_done = ['lon', 'lat', 'elevation', 'sampling_height', 'sampling_strategy', 'stime', 'etime', 'counter', 'id']
    values_done = [done_lons, done_lats, done_elevations, done_sampling_heights, done_sampling_strategies, done_stimes, done_etimes, done_counter, done_site_names]
    data_done = {keys_done[i]:values_done[i] for i in range(len(keys_done))}

    for variable in data_vars:
        data[variable] = np.zeros(lons.shape, dtype=np.float64)
        data_done[variable] = np.empty(N_points, dtype=np.float64)

    return data, data_done # Return the dicts with the data


def read_in_points_cif(comm, tree, decomp_domain, clon, hhl, number_of_NN, path_to_file, start_model, end_model, data_vars):
    """! Read in the local stationary monitoring stations on each PE in the own domain and return all of the relevant data needed for computation as dictionaries
    @param comm                 MPI communicator containing all working PE's
    @param tree                 A tree with the cells in them that you can query
    @param decomp_domain        Array with information about which cells are in this PE's prognostic area
    @param clon                 Array with the cell longitudes
    @param hhl                  Array with the height of the half levels
    @param number_of_NN         Number of nearest cells over which should be interpolated
    @param path_to_file         The path of the input file
    @param start_model          datetime object depicting the start datetime of the experiment
    @param end_model            datetime object depicting the end datetime of the experiment
    @param data_vars            The data dictionary with the data from the simulation
    @return  Tuple of dicts that symbolize the data to do and the data done
    """
    if comm.Get_rank() == 0:
        df = pd.read_csv(path_to_file, sep=',')
        df = df.dropna(subset=['lon', 'lat', 'tstep', 'station', 'alt'])
        
        df['datetime'] = pd.to_datetime(df['tstep'], unit='h', origin = start_model)

        # convert the needed data to numpy arrays
        lons = df['lon'].to_numpy()
        lats = df['lat'].to_numpy()
        stimes = df['datetime'].to_numpy()
        etimes = df['datetime'].to_numpy()
        elevations = df['alt'].to_numpy()
        sampling_heights = df['sampling_height'].to_numpy()
        sampling_strategies = df['flags'].to_numpy() + 2
        site_names = df['station']
        
        # Only keep the points, where the time is between model start and model end
        valid_mask = (stimes >= start_model) & (etimes <= end_model)
        lons = lons[valid_mask]
        lats = lats[valid_mask]
        elevations = elevations[valid_mask]
        sampling_heights = sampling_heights[valid_mask]
        sampling_strategies = sampling_strategies[valid_mask]
        site_names = site_names[valid_mask]
        stimes = stimes[valid_mask]
        etimes = etimes[valid_mask]
    else:
        lons = None
        lats = None
        elevations = None
        sampling_heights = None
        sampling_strategies = None
        site_names = None
        stimes = None
        etimes = None

    
    # Broadcast the data to all processes, from root 0
    lons = comm.bcast(lons, root = 0)
    lats = comm.bcast(lats, root = 0)
    elevations = comm.bcast(elevations, root = 0)
    sampling_heights = comm.bcast(sampling_heights, root = 0)
    sampling_strategies = comm.bcast(sampling_strategies, root = 0)
    site_names = comm.bcast(site_names, root = 0)
    stimes = comm.bcast(stimes, root = 0)
    etimes = comm.bcast(etimes, root = 0)

     # Find all of the monitoring stations in this local PE's domain and save all relevant data
    (jc_loc, jb_loc, vertical_indices_nearest, vertical_indices_second, vertical_weights,  horizontal_weights, 
        lons, lats, elevations, sampling_heights, sampling_strategies, ids, stimes, etimes) = find_points(
        lons, lats, sampling_heights, elevations, sampling_strategies, tree, decomp_domain, clon, hhl, number_of_NN, site_names, stimes, etimes, comm)

    N_points = lons.shape[0]

    # Initialize all needed arrays as empty arrays of correct size and type
    done_lons = np.empty(N_points, dtype=np.float64)
    done_lats = np.empty(N_points, dtype=np.float64)
    done_elevations = np.empty(N_points, dtype=np.float64)
    done_sampling_heights = np.empty(N_points, dtype=np.float64)
    done_sampling_strategies = np.empty(N_points, dtype=np.int32)
    done_site_names = np.empty(N_points, dtype='U20')
    done_stimes = np.empty(N_points, dtype='datetime64[ns]')
    done_etimes = np.empty(N_points, dtype='datetime64[ns]')
    

    done_counter = 0 # counter of how many of the points are already done (since last writeout)

    # Create Dicts with all of the data needed
    number_of_timesteps = np.zeros(N_points, dtype=np.int32)
    keys = ['lon', 'lat','elevation', 'sampling_height', 'sampling_strategy', 'jc_loc', 'jb_loc', 'vertical_index1', 'vertical_index2', 'vertical_weight', 'horizontal_weight', 'number_of_steps', 'id', 'stime', 'etime']
    values = [lons, lats, elevations, sampling_heights, sampling_strategies, jc_loc, jb_loc, vertical_indices_nearest, vertical_indices_second, vertical_weights, horizontal_weights, number_of_timesteps, ids, stimes, etimes]
    data = {keys[i]:values[i] for i in range(len(keys))}

    keys_done = ['lon', 'lat', 'elevation', 'sampling_height', 'sampling_strategy', 'stime', 'etime', 'counter', 'id']
    values_done = [done_lons, done_lats, done_elevations, done_sampling_heights, done_sampling_strategies, done_stimes, done_etimes, done_counter, done_site_names]
    data_done = {keys_done[i]:values_done[i] for i in range(len(keys_done))}

    for variable in data_vars:
        data[variable] = np.zeros(lons.shape, dtype=np.float64)
        data_done[variable] = np.empty(N_points, dtype=np.float64)

    return data, data_done # Return the dicts with the data

def tracking_points(datetime, data_to_do, data_done, data_np, dict_vars, operations_dict):
    """! Track the chosen variables on the chosen locations and times. Move data that is done being measured to the data_done dictionary
    @param datetime             Current datetime (np.datetime object)
    @param data_to_do           The dictionary with the data that needs to be done
    @param data_done            The dictionary with the data that is done
    @param data_np              Dictionary containing the variable names (user defined) and the data from the current timestep as numpy arrays
    @param dict_vars            Dictionary containing the variables, user defined
    @param operations_dict      Dictionary mapping the names of the signs field in the previous dict to operator. objects
    """
    if data_to_do['lon'].size > 0: # Checks if there is still work to do
        
        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is in the hour before the output of the measurement. They are ready for measurement
        measuring_mask = (
            (((data_to_do['sampling_strategy'] == 1) | (data_to_do['sampling_strategy'] == 2)) &
            (data_to_do['stime'] <= model_time_np) & 
            (data_to_do['etime'] >= model_time_np)) |
            (((data_to_do['sampling_strategy'] == 3) | (data_to_do['sampling_strategy'] == 4)) &
            (data_to_do['etime'] <= model_time_np))
        )
        if np.any(measuring_mask):
            # Filter arrays for ready stations
            jc_ready = data_to_do['jc_loc'][measuring_mask]
            jb_ready = data_to_do['jb_loc'][measuring_mask]
            vi_ready1 = data_to_do['vertical_index1'][measuring_mask]
            vi_ready2 = data_to_do['vertical_index2'][measuring_mask]
            weights_vertical_ready = data_to_do['vertical_weight'][measuring_mask]
            weights_ready = data_to_do['horizontal_weight'][measuring_mask]
            
            # For each variable make the computation specified by the dict_vars
            for variable, list in data_np.items():
                monitoring_1 = list[0][jc_ready, vi_ready1, jb_ready, 0, 0] * dict_vars[variable]['factor'][0]
                monitoring_2 = list[0][jc_ready, vi_ready2, jb_ready, 0, 0] * dict_vars[variable]['factor'][0]
                for sign, i in zip(dict_vars[variable]['signs'], range(1, len(list))):
                    monitoring_1 = operations_dict[sign](monitoring_1, list[i][jc_ready, vi_ready1, jb_ready, 0, 0] * dict_vars[variable]['factor'][i])
                    monitoring_2 = operations_dict[sign](monitoring_2, list[i][jc_ready, vi_ready2, jb_ready, 0, 0] * dict_vars[variable]['factor'][i])
                
                # Do the vertical interpolation
                monitoring_combined = monitoring_1 + weights_vertical_ready * (monitoring_2 - monitoring_1)

                # Do the horizontal interpolation
                if weights_ready.size > 0 and monitoring_combined.size > 0:
                    data_to_do[variable][measuring_mask] += np.sum(weights_ready * monitoring_combined, axis=1)
            
            data_to_do['number_of_steps'][measuring_mask] += 1


        done_mask = data_to_do['etime'] <= model_time_np # This data is done being monitored and can be output
        done_counter = data_done['counter']
        num_ready = np.sum(done_mask) # Count how many points are done
        if num_ready > 0:
        # Add all of the done points to the done arrays
            keys_done = ['lon', 'lat', 'elevation', 'sampling_height', 'sampling_strategy', 'stime', 'etime', 'counter', 'id']
            data_done['lon'][done_counter:done_counter + num_ready] = data_to_do['lon'][done_mask]
            data_done['lat'][done_counter:done_counter + num_ready] = data_to_do['lat'][done_mask]
            data_done['elevation'][done_counter:done_counter + num_ready] = data_to_do['elevation'][done_mask]
            data_done['sampling_height'][done_counter:done_counter + num_ready] = data_to_do['sampling_height'][done_mask]
            data_done['sampling_strategy'][done_counter:done_counter + num_ready] = data_to_do['sampling_strategy'][done_mask]
            data_done['stime'][done_counter:done_counter + num_ready] = data_to_do['stime'][done_mask]
            data_done['etime'][done_counter:done_counter + num_ready] = data_to_do['etime'][done_mask]
            data_done['id'][done_counter:done_counter + num_ready] = data_to_do['id'][done_mask]

            # Averaging of the data, as before we just added up contributions
            # Current behaviour is that if there are no steps made, I divide by 0 which results in NaN, which is actually great as I then in post processing know for which points I dont have data
            for variable in dict_vars:
                data_done[variable][done_counter:done_counter + num_ready] = data_to_do[variable][done_mask] / data_to_do['number_of_steps'][done_mask]


            # Keep count of how many points are done
            data_done['counter'] += num_ready

            # Only keep the points that aren't done yet
            keep_mask = ~done_mask

            # Filter the entrys in the dictionary and only keep the points that aren't done yet
            for key in data_to_do:
                data_to_do[key] = data_to_do[key][keep_mask]
        
