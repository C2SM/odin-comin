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
import os
from scipy.spatial import KDTree
import datetime as datumzeit

first_write_done_monitoring = False # Help to know if output file already exists
first_write_done_single = False
debug = True
jg = 1 # we do compututations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
singlepoint_monitoring = []
# singlepoint_monitoring = [[26.0, 46.0, 0.0, True, '2019-01-01T00:01:00.000'], [23.0, 47.0, 0.0, True, '2019-01-01T01:02:00.000']] # first coordinate is longitude, second coordinate is latitude, third is height in meters and 4 indicates if it's height above ground (true) or height above sea (false), 5 indicates the timestep the measurement was done
monitoring_stations = [[26.0, 46.0, 0.0, True], [23.0, 47.0, 0.0, True]] # first coordinate is longitude, second coordinate is latitude, third is height in meters and 4 indicates if it's height above ground (true) or height above sea (false)
time_interval_writeout = 900 # in seconds


def message(message_string, rank):
    """Short helper function to print a message on one PE"""
    if (comin.parallel_get_host_mpi_rank() == rank):
        print(f"ComIn point_source.py: {message_string}", file=sys.stderr)

def lonlat2xyz(lon, lat):
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

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
    global local_monitoring_stations, number_of_timesteps, local_time_instances, done_local_time_instances, singl

    # all arrays are for domain 1 only
    domain = comin.descrdata_get_domain(jg)
    clon = np.asarray(domain.cells.clon)
    clat = np.asarray(domain.cells.clat)
    hhl = np.asarray(domain.cells.hhl)

    xyz = np.c_[lonlat2xyz(clon.ravel(),clat.ravel())]
    decomp_domain = np.asarray(domain.cells.decomp_domain)

    number_of_timesteps = 0
    local_monitoring_stations = []
    local_time_instances = []
    done_local_time_instances = None

    tree = KDTree(xyz)
    for i in range(len(monitoring_stations)):
        station_lon = monitoring_stations[i][0]
        station_lat = monitoring_stations[i][1]
        station_height = monitoring_stations[i][2]
        above_ground = monitoring_stations[i][3]

        dd, ii = tree.query([lonlat2xyz(np.deg2rad(station_lon), np.deg2rad(station_lat))], k=1)

        if (decomp_domain.ravel()[ii] == 0):
            # point found is inside prognostic area
            # This implicitly assumes that on each other PE, the nearest neighbor is located in the halo zone
            jc_loc, jb_loc = np.unravel_index(ii, clon.shape)

            local_hhl = hhl[jc_loc, :, jb_loc].squeeze()
            h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])
            
            height_above_sea = station_height
            if(above_ground):
                height_above_sea += local_hhl[-1]
            
            vertical_index = int(np.argmin(np.abs(h_mid - height_above_sea)))
            
            # actual_height = h_mid[vertical_index].item()
            
            local_monitoring_stations.append({
                'jc_loc': jc_loc,
                'jb_loc': jb_loc,
                'current_CH4': 0,
                'height': station_height,
                'vertical_index': vertical_index,
                'lon': station_lon,
                'lat': station_lat
            })

    for i in range(len(singlepoint_monitoring)):
        station_lon = singlepoint_monitoring[i][0]
        station_lat = singlepoint_monitoring[i][1]
        station_height = singlepoint_monitoring[i][2]
        above_ground = singlepoint_monitoring[i][3]
        timepoint = singlepoint_monitoring[i][4]

        dd, ii = tree.query([lonlat2xyz(np.deg2rad(station_lon), np.deg2rad(station_lat))], k=1)

        if (decomp_domain.ravel()[ii] == 0):
            # point found is inside prognostic area
            # This implicitly assumes that on each other PE, the nearest neighbor is located in the halo zone
            jc_loc, jb_loc = np.unravel_index(ii, clon.shape)

            local_hhl = hhl[jc_loc, :, jb_loc].squeeze()
            h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])
            
            height_above_sea = station_height
            if(above_ground):
                height_above_sea += local_hhl[-1]
            
            vertical_index = int(np.argmin(np.abs(h_mid - height_above_sea)))
            
            # actual_height = h_mid[vertical_index].item()
            
            local_time_instances.append({
                'jc_loc': jc_loc,
                'jb_loc': jb_loc,
                'CH4': 0,
                'height': station_height,
                'vertical_index': vertical_index,
                'timepoint': timepoint, 
                'lon': station_lon,
                'lat': station_lat
            })

@comin.register_callback(comin.EP_ATM_TIMELOOP_START)
def input_flight_data():
    """reading in csv flight file"""
    global debug, local_time_instances, done_local_time_instances, singlepoint_monitoring

    datetime = comin.current_get_datetime()

    if(pd.to_datetime(datetime).time() == datumzeit.time(0,0)):
        comm = MPI.COMM_WORLD
        newGroup = comm.group.Excl([123, 124, 125, 126, 127])
        newComm = comm.Create_group(newGroup)
        rank = newComm.Get_rank()
        if debug:
            if rank == 0:
                df = pd.read_csv('flight.csv', sep=';')

                df.columns = [col.strip() for col in df.columns]
                df = df.dropna(subset=['Time_EPOCH', 'AGL_m', 'Longitude', 'Latitude'])
                df['datetime'] = pd.to_datetime(df['Time_EPOCH'], unit='s')



                target_start = datumzeit.datetime(2019, 1, 1, 0, 1, 0)
                original_start = df['datetime'].min()
                delta = original_start - target_start
                df['datetime'] = df['datetime'] - delta

                df['timestamp'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S.000')

                singlepoint_monitoring = [
                    [row['Longitude'], row['Latitude'], row['AGL_m'], True, row['timestamp']]
                    for _, row in df.iterrows()
                ]

            singlepoint_monitoring = newComm.bcast(singlepoint_monitoring, root=0)

        # all arrays are for domain 1 only
        domain = comin.descrdata_get_domain(jg)
        clon = np.asarray(domain.cells.clon)
        clat = np.asarray(domain.cells.clat)
        hhl = np.asarray(domain.cells.hhl)

        xyz = np.c_[lonlat2xyz(clon.ravel(),clat.ravel())]
        decomp_domain = np.asarray(domain.cells.decomp_domain)

        local_time_instances = []
        done_local_time_instances = None

        tree = KDTree(xyz)
        for i in range(len(singlepoint_monitoring)):
            station_lon = singlepoint_monitoring[i][0]
            station_lat = singlepoint_monitoring[i][1]
            station_height = singlepoint_monitoring[i][2]
            above_ground = singlepoint_monitoring[i][3]
            timepoint = singlepoint_monitoring[i][4]

            dd, ii = tree.query([lonlat2xyz(np.deg2rad(station_lon), np.deg2rad(station_lat))], k=1)

            if (decomp_domain.ravel()[ii] == 0):
                # point found is inside prognostic area
                # This implicitly assumes that on each other PE, the nearest neighbor is located in the halo zone
                jc_loc, jb_loc = np.unravel_index(ii, clon.shape)

                local_hhl = hhl[jc_loc, :, jb_loc].squeeze()
                h_mid = 0.5 * (local_hhl[:-1] + local_hhl[1:])
                
                height_above_sea = station_height
                if(above_ground):
                    height_above_sea += local_hhl[-1]
                
                vertical_index = int(np.argmin(np.abs(h_mid - height_above_sea)))
                
                # actual_height = h_mid[vertical_index].item()
                
                local_time_instances.append({
                    'jc_loc': jc_loc,
                    'jb_loc': jb_loc,
                    'CH4': 0,
                    'height': station_height,
                    'vertical_index': vertical_index,
                    'timepoint': timepoint, 
                    'lon': station_lon,
                    'lat': station_lat
                })
        debug = False

@comin.register_callback(comin.EP_ATM_TIMELOOP_END) # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
def tracking_CH4_total():
    """tracking of CH4 Emissions"""
    global number_of_timesteps, local_time_instances, done_local_time_instances
    dtime = comin.descrdata_get_timesteplength(jg)
    datetime = comin.current_get_datetime() # This could maybe be useful for later, example for format: 2019-01-01T00:01:00.000
    number_of_timesteps += 1 # tracking number of steps, to in the end average over the correct time

    # Convert all of them to numpy arrays
    CH4_EMIS_np = np.asarray(CH4_EMIS)
    CH4_BG_np = np.asarray(CH4_BG)
    
    for i in range(len(local_monitoring_stations)):
        height = local_monitoring_stations[i]['height']
        vertical_index = local_monitoring_stations[i]['vertical_index']
        jc_loc = local_monitoring_stations[i]['jc_loc']
        jb_loc = local_monitoring_stations[i]['jb_loc']

        # This is the main summation of all of the CH4 sources
        local_monitoring_stations[i]['current_CH4'] += CH4_EMIS_np[jc_loc, vertical_index, jb_loc, 0, 0] * 1e9
        local_monitoring_stations[i]['current_CH4'] += CH4_BG_np[jc_loc, vertical_index, jb_loc, 0, 0]

    new_local_time_instances = []
    for i in range(len(local_time_instances)):
        timepoint = local_time_instances[i]['timepoint']
        if pd.to_datetime(timepoint) <= pd.to_datetime(datetime): 
            if done_local_time_instances is None:
                done_local_time_instances = []
            height = local_time_instances[i]['height']
            vertical_index = local_time_instances[i]['vertical_index']
            jc_loc = local_time_instances[i]['jc_loc']
            jb_loc = local_time_instances[i]['jb_loc']
            lon = local_time_instances[i]['lon']
            lat = local_time_instances[i]['lat']

            # This is the main summation of all of the CH4 sources
            CH4_current = CH4_EMIS_np[jc_loc, vertical_index, jb_loc, 0, 0] * 1e9 + CH4_BG_np[jc_loc, vertical_index, jb_loc, 0, 0]

            done_local_time_instances.append({
                'CH4': float(CH4_current),
                'height': float(height),
                'timepoint': timepoint, 
                'lon': lon,
                'lat': lat
            })
        else:
            new_local_time_instances.append(local_time_instances[i])
    local_time_instances = new_local_time_instances


    elapsed_time = dtime * number_of_timesteps
    # Now this is where we log the averaged CH4, it is done by gathering the local monitoring stations and then storing them in an xarray, also writing them out then into an outputfile
    if (elapsed_time >= time_interval_writeout):
        local_data_monitoring = None
        for i in range(len(local_monitoring_stations)):
            if(i==0):
                local_data_monitoring = []
            avg_CH4 = local_monitoring_stations[i]['current_CH4'] / (number_of_timesteps)
            lon = local_monitoring_stations[i]['lon']
            lat = local_monitoring_stations[i]['lat']
            height = local_monitoring_stations[i]['height']
            local_data_monitoring.append( {'time_period': elapsed_time, 'avg_CH4': avg_CH4, 'datetime': datetime, 'lon': lon, 'lat': lat, 'height': height})
            local_monitoring_stations[i]['current_CH4'] = 0
        
        comm = MPI.COMM_WORLD
        newGroup = comm.group.Excl([123, 124, 125, 126, 127])
        newComm = comm.Create_group(newGroup)
        rank = newComm.Get_rank()
        gathered_monitoring = newComm.gather(local_data_monitoring, root = 0)
        gathered_instances = newComm.gather(done_local_time_instances, root = 0)


        if newComm.Get_rank() == 0:
            gathered_monitoring = [d for d in gathered_monitoring if d is not None]
            gathered_instances = [d for d in gathered_instances if d is not None]
            flattened_monitoring = [entry for sublist in gathered_monitoring for entry in sublist]

            from collections import defaultdict

            station_data = defaultdict(list)
            time_label = None

            for entry in flattened_monitoring:
                station_key = (entry["lon"], entry["lat"], entry['height'])
                station_data[station_key].append(entry["avg_CH4"].item())
                time_label = entry["datetime"]

            station_keys = sorted(station_data.keys())

            lons = [lon for lon, lat, height in station_keys]
            lats = [lat for lon, lat, height in station_keys]
            heights = [height for lon, lat, height in station_keys]
            station_ids = [f"station_{i}" for i in range(len(station_keys))]
            avg_ch4 = [station_data[key][0] for key in station_keys]

            ds = xr.Dataset(
                {
                    "avg_CH4": (["station", "time"], np.array(avg_ch4)[..., np.newaxis]),
                },
                coords={
                    "station": station_ids,
                    "lon": ("station", lons),
                    "lat": ("station", lats),
                    "height": ("station", heights),
                    "time": [pd.to_datetime(time_label)],
                },
            )

            ds["avg_CH4"].attrs["units"] = "ppb"
            ds["avg_CH4"].attrs["long_name"] = "Average CH4 concentration"
            ds["lon"].attrs["units"] = "degrees_east"
            ds["lat"].attrs["units"] = "degrees_north"
            ds["height"].attrs["units"] = "m"
            ds["time"].attrs["standard_name"] = "time"
            # ds["time"].attrs["units"] = "seconds since 2019-01-01 00:00:00"
            # ds["time"].attrs["calendar"] = "proleptic_gregorian"

            # encoding = {
            #     "time": {}
            # }
            encoding = {
                "time": {
                    "units": "seconds since 2019-01-01 00:00:00",
                    "calendar": "proleptic_gregorian"
                }
            }

            output_file = "tracked_ch4.nc"

            global first_write_done_monitoring
            if not first_write_done_monitoring:
                ds.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")
                first_write_done_monitoring = True
            else:
                existing_ds = xr.open_dataset(output_file)
                combined = xr.concat([existing_ds, ds], dim="time")
                existing_ds.close()
                combined.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")


            
            ## Now the local time instances:
            csv_file = "ch4_flight.csv"
            if len(gathered_instances) != 0:

                global first_write_done_single
                flattened_instances = [item for sublist in gathered_instances for item in sublist]

                df = pd.DataFrame(flattened_instances)

                df.to_csv(csv_file, mode='a', header=not first_write_done_single, index=False)
                if not first_write_done_single:
                    first_write_done_single = True


        number_of_timesteps = 0
        done_local_time_instances = None
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def CH4_destructor():
    message("CH4_destructor called!", msgrank)
