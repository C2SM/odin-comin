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
import datetime as datetimelib
import os

## Constants:
NUMBER_OF_NN = 4 # Number of nearest neighbour cells over which should be interpolated
N_COMPUTE_PES = 123  # number of compute PEs
jg = 1 # we do compututations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
# days_of_flights = [7, 8]
days_of_flights = [[2019, 10, 7], [2019, 10, 8]] # Up until now manually put in the dates of the flights
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

monitoring_lons = np.array([26.0, 23.0]) # predefine the monitoring stations. This could in future also be done via file inread
monitoring_lats = np.array([46.0, 47.0])
monitoring_heights = np.array([0.0, 0.0])
monitoring_is_abg = np.array([True, True])


def message(message_string, rank):
    """Short helper function to print a message on one PE"""
    if (comin.parallel_get_host_mpi_rank() == rank):
        print(f"ComIn point_source.py: {message_string}", file=sys.stderr)

def lonlat2xyz(lon, lat):
    """Short helper function for calculating xyz coordinates from longitues and latitudes"""
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

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
            
            # Csv filename, will maybe change later, when we know how the flight data csv's are named
            csv_file = "flight_modeled" + str(pd.to_datetime(datetime).day) + ".csv"

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
    global CH4_EMIS, CH4_BG
    entry_points = [comin.EP_ATM_TIMELOOP_END] # TIMELOOP END is atm selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
    
    # Request to get the wanted variables (i.e. the EMIS and the BG). We only want to read the data, not write
    CH4_EMIS = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
    CH4_BG = comin.var_get(entry_points, ("CH4_BG", jg), comin.COMIN_FLAG_READ)

    message("data_constructor successful", msgrank)

@comin.register_callback(comin.EP_ATM_INIT_FINALIZE)
def stations_init():
    global number_of_timesteps, clon, clat, hhl, xyz, decomp_domain, tree # variables with domain info, and general information
    # All of the monitoring variables
    global jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring1, vertical_indices_monitoring2, vertical_weight_monitoring, current_CH4_monitoring
    global monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg, weights_monitoring
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

    # get the datetime from comin
    datetime = comin.current_get_datetime()

    # Currently we read in at midnight for the following night
    if(pd.to_datetime(datetime).time() == datetimelib.time(0,0)):
        day = pd.to_datetime(datetime).day # get the current day for the name of the filepath
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
                flight_file = 'flight' + str(day) + '.csv'
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
        done_times = np.empty(N_flight_points)
        done_CH4 = np.empty(N_flight_points, dtype=np.float64)

        done_counter = 0 # counter of how many of the N_flight_points are already done (This day)

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

    dtime = comin.descrdata_get_timesteplength(jg) # size of every timestep 
    datetime = comin.current_get_datetime() # get datetime info. example for format: 2019-01-01T00:01:00.000
    number_of_timesteps += 1 # tracking number of steps, to in the end average after the correct time and average of the correct amount of timesteps

    # Convert all of them to numpy arrays
    CH4_EMIS_np = np.asarray(CH4_EMIS)
    CH4_BG_np = np.asarray(CH4_BG)

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

    

    ## Writeout
    elapsed_time = dtime * number_of_timesteps # Time that has elapsed since last writeout (or since start if there was no writeout yet)
    if (elapsed_time >= time_interval_writeout): # If we are above the time intervall writeout defined on the very top, we want to write out
        write_monitoring_stations(datetime) # Write out the monitoring stations
        write_singlepoints(datetime) # Write out the flight data single points

        # Reset data
        number_of_timesteps = 0
        done_counter = 0
        current_CH4_monitoring[:] = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def CH4_destructor():
    message("CH4_destructor called!", msgrank)
