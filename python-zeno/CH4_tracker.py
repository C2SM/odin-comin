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
import datetime as datumzeit
import os

first_write_done_monitoring = False # Help to know if output file already exists
# first_write_done_single = False
singlepoint_done = False
# debug = True
NUMBER_OF_NN = 4 # Number of nearest niehgbour cells over which should be interpolated
N_COMPUTE_PES = 123  # number of compute PEs
jg = 1 # we do compututations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
days_of_flights = [7, 8]

singlepoint_lons = np.array([]) # predefine the arrays as empty
singlepoint_lats = np.array([])
singlepoint_heights = np.array([])
singlepoint_is_abg = np.array([])
singlepoint_timestep = np.array([])

monitoring_lons = np.array([26.0, 23.0]) # predefine the monitoring stations. This could in future also be done via file inread
monitoring_lats = np.array([46.0, 47.0])
monitoring_heights = np.array([0.0, 0.0])
monitoring_is_abg = np.array([True, True])


time_interval_writeout = 900 # in seconds


def message(message_string, rank):
    """Short helper function to print a message on one PE"""
    if (comin.parallel_get_host_mpi_rank() == rank):
        print(f"ComIn point_source.py: {message_string}", file=sys.stderr)

def lonlat2xyz(lon, lat):
    """Short helper function for calculating xyz coordinates"""
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

def find_stations_monitor(lons, lats, heights, are_abg, tree, decomp_domain, clon, hhl):
    """Find the local stationary monitoring stations on each PE in the own domain and return all of the relevant data needed for computation"""
    jc_locs = []
    jb_locs = []
    vertical_indices = []
    weights_all = []
    lons_local = []
    lats_local = []
    heights_local = []
    are_abg_local = []

    for lon, lat, height, above_ground in zip(lons, lats, heights, are_abg):
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = NUMBER_OF_NN)

        if decomp_domain.ravel()[ii[0][0]] == 0:
            jc_loc, jb_loc = np.unravel_index(ii[0], clon.shape)

            # create a mask: which neighbors are local
            local_mask = (decomp_domain.ravel()[ii[0]] == 0)

            # filter jc and jb
            jc_loc = jc_loc[local_mask]
            jb_loc = jb_loc[local_mask]
            dd_local = dd[0][local_mask] 


            jc_row = []
            jb_row = []
            vertical_row = []
            weight_row = []

            for jc, jb in zip(jc_loc, jb_loc):
                local_hhl = hhl[jc, :, jb].squeeze()
                h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])

                height_above_sea = height
                if above_ground:
                    height_above_sea += local_hhl[-1]

                vertical_index = int(np.argmin(np.abs(h_mid - height_above_sea)))
                jc_row.append(jc)
                jb_row.append(jb)
                vertical_row.append(vertical_index)

            if np.any(dd_local==0):
                print('The longitude/latitude coincides identically with an ICON cell, which is an issue for the inverse distance weighting.', file=sys.stderr)
                print('I will slightly modify this value to avoid errors.', file=sys.stderr)
                dd_local[dd_local==0] = 1e-12
            weights = 1.0 / dd_local
            weights = weights / np.sum(weights)

            weight_row = weights.tolist()
            # If fewer than NUMBER_OF_NN neighbors, pad with -1
            while len(jc_row) < NUMBER_OF_NN:
                jc_row.append(-1)
                jb_row.append(-1)
                vertical_row.append(-1)
                weight_row.append(0.0)

            jc_locs.append(jc_row)
            jb_locs.append(jb_row)
            vertical_indices.append(vertical_row)
            weights_all.append(weight_row)
            lons_local.append(lon)
            lats_local.append(lat)
            heights_local.append(height)
            are_abg_local.append(above_ground)

    return (np.array(jc_locs, dtype=np.int32),
            np.array(jb_locs, dtype=np.int32),
            np.array(vertical_indices, dtype=np.int32),
            np.array(weights_all, dtype=np.float64),
            np.array(lons_local),
            np.array(lats_local),
            np.array(heights_local),
            np.array(are_abg_local))

def find_stations_singlepoint(lons, lats, heights, are_abg, timesteps, tree, decomp_domain, clon, hhl):
    """Find the local monitoring points that should be read out on a single timestep on each PE in the own domain and return all of the relevant data needed for computation"""
    jc_locs = []
    jb_locs = []
    vertical_indices = []
    weights_all = []
    lons_local = []
    lats_local = []
    heights_local = []
    are_abg_local = []
    timesteps_local = []

    for lon, lat, height, above_ground, timestep in zip(lons, lats, heights, are_abg, timesteps):
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = NUMBER_OF_NN)

        if decomp_domain.ravel()[ii[0][0]] == 0:
            jc_loc, jb_loc = np.unravel_index(ii[0], clon.shape)

            # create a mask: which neighbors are local
            local_mask = (decomp_domain.ravel()[ii[0]] == 0)

            # filter jc and jb
            jc_loc = jc_loc[local_mask]
            jb_loc = jb_loc[local_mask]
            dd_local = dd[0][local_mask] 

            jc_row = []
            jb_row = []
            vertical_row = []
            weight_row = []

            for jc, jb in zip(jc_loc, jb_loc):
                local_hhl = hhl[jc, :, jb].squeeze()
                h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])

                height_above_sea = height
                if above_ground:
                    height_above_sea += local_hhl[-1]

                vertical_index = int(np.argmin(np.abs(h_mid - height_above_sea)))
                jc_row.append(jc)
                jb_row.append(jb)
                vertical_row.append(vertical_index)

            if np.any(dd_local==0):
                print('The longitude/latitude coincides identically with an ICON cell, which is an issue for the inverse distance weighting.', file=sys.stderr)
                print('I will slightly modify this value to avoid errors.', file=sys.stderr)
                dd_local[dd_local==0] = 1e-12           
            weights = 1.0 / dd_local
            weights = weights / np.sum(weights)

            weight_row = weights.tolist()
            # If fewer than NUMBER_OF_NN neighbors, pad with -1
            while len(jc_row) < NUMBER_OF_NN:
                jc_row.append(-1)
                jb_row.append(-1)
                vertical_row.append(-1)
                weight_row.append(0.0)

            jc_locs.append(jc_row)
            jb_locs.append(jb_row)
            vertical_indices.append(vertical_row)
            weights_all.append(weight_row)
            lons_local.append(lon)
            lats_local.append(lat)
            heights_local.append(height)
            are_abg_local.append(above_ground)
            timesteps_local.append(timestep)

    return (np.array(jc_locs, dtype=np.int32),
            np.array(jb_locs, dtype=np.int32),
            np.array(vertical_indices, dtype=np.int32),
            np.array(weights_all, dtype=np.float64),
            np.array(lons_local),
            np.array(lats_local),
            np.array(heights_local),
            np.array(are_abg_local),
            np.array(timesteps_local))

def write_singlepoints(datetime):
    """Function to writeout the single timepoint data"""
    global done_counter, comm # first_write_done_single

    done_data_local = None
    if done_counter > 0:
        done_data_local = {
            "lon": done_lons[:done_counter],
            "lat": done_lats[:done_counter],
            "height": done_heights[:done_counter],
            "timepoint": done_times[:done_counter],
            "CH4": done_CH4[:done_counter],
        }

    gathered_done_data = comm.gather(done_data_local, root=0)

    if rank == 0:
        # Flatten
        final_data = {
            "lon": [],
            "lat": [],
            "height": [],
            "timepoint": [],
            "CH4": [],
        }
        for d in gathered_done_data:
            if d is not None:
                for key in final_data:
                    final_data[key].append(d[key])

        # Check if there is anything to write
        if any(len(lst) > 0 for lst in final_data.values()):
            for key in final_data:
                final_data[key] = np.concatenate(final_data[key])

            df = pd.DataFrame(final_data)
            

            # Write to CSV
            csv_file = "flight_modeled" + str(pd.to_datetime(datetime).day) + ".csv"
            
            file_exists = os.path.isfile(csv_file)
            df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

            # if not first_write_done_single:
            #     first_write_done_single = True

def write_monitoring_stations(datetime):
    """Function to writeout the stationary monitoring stations"""
    global first_write_done_monitoring, comm

    # Calculate averaged CH4
    avg_CH4_local = current_CH4_monitoring / number_of_timesteps
    avg_CH4_local = np.asarray(avg_CH4_local).ravel()

    # Gather everything
    gathered_avg_CH4 = comm.gather(avg_CH4_local, root=0)
    gathered_lons = comm.gather(monitoring_lons, root=0)
    gathered_lats = comm.gather(monitoring_lats, root=0)
    gathered_heights = comm.gather(monitoring_heights, root=0)

    if rank == 0:
        avg_CH4_flat = np.concatenate(gathered_avg_CH4)
        lons_flat = np.concatenate(gathered_lons)
        lats_flat = np.concatenate(gathered_lats)
        heights_flat = np.concatenate(gathered_heights)

        station_ids = [f"station_{i}" for i in range(avg_CH4_flat.shape[0])]

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

        ds["avg_CH4"].attrs["units"] = "ppb"
        ds["avg_CH4"].attrs["long_name"] = "Average CH4 concentration"
        ds["lon"].attrs["units"] = "degrees_east"
        ds["lat"].attrs["units"] = "degrees_north"
        ds["height"].attrs["units"] = "m"
        ds["time"].attrs["standard_name"] = "time"

        encoding = {
            "time": {
                "units": "seconds since 2019-01-01 00:00:00",
                "calendar": "proleptic_gregorian"
            }
        }

        output_file = "tracked_ch4.nc"
        if not first_write_done_monitoring:
            ds.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")
            del ds
            first_write_done_monitoring = True
        else:
            existing_ds = xr.open_dataset(output_file)
            combined = xr.concat([existing_ds, ds], dim="time")
            existing_ds.close()
            combined.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")
            combined.close()
            del ds


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def data_constructor():
#     """Constructor: Get pointers to data"""
    global CH4_EMIS, CH4_BG
    entry_points = [comin.EP_ATM_TIMELOOP_END] # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
    
    CH4_EMIS = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
    CH4_BG = comin.var_get(entry_points, ("CH4_BG", jg), comin.COMIN_FLAG_READ)

    message("data_constructor successful", msgrank)

@comin.register_callback(comin.EP_ATM_INIT_FINALIZE)
def stations_init():
    global number_of_timesteps, clon, clat, hhl, xyz, decomp_domain, tree, jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring, current_CH4_monitoring, monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg, comm, rank, weights_monitoring
    world_comm = MPI.COMM_WORLD

    group_world = world_comm.Get_group()
    group = group_world.Incl(list(range(N_COMPUTE_PES)))
    comm = world_comm.Create_group(group)

    if comm != MPI.COMM_NULL:
        rank = comm.Get_rank()
    else:
        rank = None
    
    # all arrays are for domain 1 only
    domain = comin.descrdata_get_domain(jg)
    clon = np.asarray(domain.cells.clon)
    clat = np.asarray(domain.cells.clat)
    hhl = np.asarray(domain.cells.hhl)

    xyz = np.c_[lonlat2xyz(clon.ravel(),clat.ravel())]
    decomp_domain = np.asarray(domain.cells.decomp_domain)

    number_of_timesteps = 0

    tree = KDTree(xyz)
    (jc_loc_monitoring, jb_loc_monitoring, vertical_indices_monitoring, weights_monitoring, 
 monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg) = find_stations_monitor(
    monitoring_lons, monitoring_lats, monitoring_heights, monitoring_is_abg,
    tree, decomp_domain, clon, hhl
)
    current_CH4_monitoring = np.zeros(monitoring_lons.shape, dtype=np.float64)

@comin.register_callback(comin.EP_ATM_TIMELOOP_START)
def input_flight_data():
    """reading in csv flight file"""
    global singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_is_abg, singlepoint_timestep, jc_loc_singlepoint, jb_loc_singlepoint, vertical_indices_singlepoint, CH4_singlepoint, done_lons, done_lats, done_heights, done_times, done_CH4, done_counter, N_flight_points, singlepoint_done, weights_singlepoint

    datetime = comin.current_get_datetime()

    if(pd.to_datetime(datetime).time() == datumzeit.time(0,0)):
        # if debug:
        day = pd.to_datetime(datetime).day
        if day in days_of_flights:
            singlepoint_lons = None
            singlepoint_lats = None
            singlepoint_heights = None
            singlepoint_is_abg = None
            singlepoint_timestep = None
            if rank == 0:
                flight_file = 'flight' + str(day) + '.csv'
                df = pd.read_csv(flight_file, sep=';')

                df.columns = [col.strip() for col in df.columns]
                df = df.dropna(subset=['Time_EPOCH', 'AGL_m', 'Longitude', 'Latitude'])
                df['datetime'] = pd.to_datetime(df['Time_EPOCH'], unit='s')



                # target_start = datumzeit.datetime(2019, 1, 1, 0, 1, 0)
                # original_start = df['datetime'].min()
                # delta = original_start - target_start
                # df['datetime'] = df['datetime'] - delta

                singlepoint_lons = df['Longitude'].to_numpy()
                singlepoint_lats = df['Latitude'].to_numpy()
                singlepoint_heights = df['AGL_m'].to_numpy()
                singlepoint_is_abg = np.ones_like(singlepoint_lons, dtype=bool)
                singlepoint_timestep = df['datetime'].to_numpy()

            singlepoint_lons = comm.bcast(singlepoint_lons, root=0)
            singlepoint_lats = comm.bcast(singlepoint_lats, root=0)
            singlepoint_heights = comm.bcast(singlepoint_heights, root=0)
            singlepoint_is_abg = comm.bcast(singlepoint_is_abg, root=0)
            singlepoint_timestep = comm.bcast(singlepoint_timestep, root=0)      


            (jc_loc_singlepoint, jb_loc_singlepoint, vertical_indices_singlepoint, weights_singlepoint, 
                singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_is_abg, singlepoint_timestep) = find_stations_singlepoint(singlepoint_lons, singlepoint_lats, singlepoint_heights, singlepoint_is_abg, singlepoint_timestep, tree, decomp_domain, clon, hhl)
        N_flight_points = singlepoint_lons.shape[0]
        CH4_singlepoint = np.empty(N_flight_points)
        done_lons = np.empty(N_flight_points, dtype=np.float64)
        done_lats = np.empty(N_flight_points, dtype=np.float64)
        done_heights = np.empty(N_flight_points, dtype=np.float64)
        done_times = np.empty(N_flight_points, dtype="datetime64[ns]")
        done_CH4 = np.empty(N_flight_points, dtype=np.float64)

        done_counter = 0 

        # debug = False
        singlepoint_done = False

@comin.register_callback(comin.EP_ATM_TIMELOOP_END) # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
def tracking_CH4_total():
    """tracking of CH4 Emissions"""
    global number_of_timesteps, current_CH4_monitoring, jc_loc_singlepoint, jb_loc_singlepoint, vertical_indices_singlepoint, CH4_singlepoint, singlepoint_timestep, singlepoint_lons, singlepoint_lats, singlepoint_heights, done_lons, done_lats, done_heights, done_times, done_CH4, done_counter, N_flight_points, singlepoint_done, weights_singlepoint
    dtime = comin.descrdata_get_timesteplength(jg)
    datetime = comin.current_get_datetime() # This could maybe be useful for later, example for format: 2019-01-01T00:01:00.000
    number_of_timesteps += 1 # tracking number of steps, to in the end average over the correct time

    # Convert all of them to numpy arrays
    CH4_EMIS_np = np.asarray(CH4_EMIS)
    CH4_BG_np = np.asarray(CH4_BG)
    # Fetch CH4 values
    CH4_monitoring_all = (
        CH4_EMIS_np[jc_loc_monitoring, vertical_indices_monitoring, jb_loc_monitoring, 0, 0] * 1e9 +
        CH4_BG_np[jc_loc_monitoring, vertical_indices_monitoring, jb_loc_monitoring, 0, 0]
    )  
    if weights_monitoring.size > 0 and CH4_monitoring_all.size > 0:
        current_CH4_monitoring += np.sum(weights_monitoring * CH4_monitoring_all, axis=1)

    model_time_np = np.datetime64(datetime)

    if not singlepoint_done and singlepoint_timestep.size > 0:
        ready_mask = singlepoint_timestep <= model_time_np
        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready = jc_loc_singlepoint[ready_mask]
            jb_ready = jb_loc_singlepoint[ready_mask]
            vi_ready = vertical_indices_singlepoint[ready_mask]
            weights_ready = weights_singlepoint[ready_mask]

            # Fetch CH4 values
            CH4_ready_all = (
                CH4_EMIS_np[jc_ready, vi_ready, jb_ready, 0, 0] * 1e9 +
                CH4_BG_np[jc_ready, vi_ready, jb_ready, 0, 0]
            )

            CH4_ready = np.sum(weights_ready * CH4_ready_all, axis=1)

            num_ready = np.sum(ready_mask)

            done_lons[done_counter:done_counter + num_ready] = singlepoint_lons[ready_mask]
            done_lats[done_counter:done_counter + num_ready] = singlepoint_lats[ready_mask]
            done_heights[done_counter:done_counter + num_ready] = singlepoint_heights[ready_mask]
            done_times[done_counter:done_counter + num_ready] = singlepoint_timestep[ready_mask]
            done_CH4[done_counter:done_counter + num_ready] = CH4_ready

            done_counter += num_ready

            keep_mask = ~ready_mask
            singlepoint_lons = singlepoint_lons[keep_mask]
            singlepoint_lats = singlepoint_lats[keep_mask]
            singlepoint_heights = singlepoint_heights[keep_mask]
            singlepoint_timestep = singlepoint_timestep[keep_mask]
            jc_loc_singlepoint = jc_loc_singlepoint[keep_mask]
            jb_loc_singlepoint = jb_loc_singlepoint[keep_mask]
            vertical_indices_singlepoint = vertical_indices_singlepoint[keep_mask]
            weights_singlepoint = weights_singlepoint[keep_mask]

        if singlepoint_timestep.size == 0:
            singlepoint_done = True 


    elapsed_time = dtime * number_of_timesteps
    # Now this is where we log the data we collected
    if (elapsed_time >= time_interval_writeout):
        write_monitoring_stations(datetime)
        write_singlepoints(datetime)

        # Reset data
        number_of_timesteps = 0
        done_counter = 0
        current_CH4_monitoring[:] = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def CH4_destructor():
    message("CH4_destructor called!", msgrank)
