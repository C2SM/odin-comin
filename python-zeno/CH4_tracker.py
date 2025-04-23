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

first_write_done = False
jg = 1 # we do compututations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
monitoring_stations = [[26.0, 46.0, 0], [23.0, 47.0, 0]] # first coordinate is longitude, second coordinate is latitude, third is height
time_interval = 3600 # in seconds


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
    global CH4_EMIS, CH4_BG # , CH4_TNO, CH4_OMV
    entry_points = [comin.EP_ATM_TIMELOOP_END] # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
    
    CH4_EMIS = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
    # CH4_TNO = comin.var_get(entry_points, ("CH4_TNO", jg), comin.COMIN_FLAG_READ)
    # CH4_OMV = comin.var_get(entry_points, ("CH4_OMV", jg), comin.COMIN_FLAG_READ)
    CH4_BG = comin.var_get(entry_points, ("CH4_BG", jg), comin.COMIN_FLAG_READ)

    message("data_constructor successful", msgrank)

@comin.register_callback(comin.EP_ATM_INIT_FINALIZE)
def stations_init():
    global local_monitoring_stations, number_of_timesteps

    # all arrays are for domain 1 only
    domain = comin.descrdata_get_domain(jg)
    clon = np.asarray(domain.cells.clon)
    clat = np.asarray(domain.cells.clat)
    xyz = np.c_[lonlat2xyz(clon.ravel(),clat.ravel())]
    decomp_domain = np.asarray(domain.cells.decomp_domain)

    number_of_timesteps = 0
    local_monitoring_stations = []

    tree = KDTree(xyz)
    for i in range(len(monitoring_stations)):
        station_lon = monitoring_stations[i][0]
        station_lat = monitoring_stations[i][1]
        station_height = monitoring_stations[i][2]
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(station_lon), np.deg2rad(station_lat))], k=1)

        # iii = ii[0]
        if (decomp_domain.ravel()[ii] == 0):
            # point found is inside prognostic area
            # This implicitly assumes that on each other PE, the nearest neighbor is located in the halo zone
            jc_loc, jb_loc = np.unravel_index(ii, clon.shape)
            message(f"Monitoring station {i} found at PE {comin.parallel_get_host_mpi_rank()}, clon={np.rad2deg(clon[jc_loc,jb_loc])}, clat={np.rad2deg(clat[jc_loc,jb_loc])}", comin.parallel_get_host_mpi_rank())
            local_monitoring_stations.append({
                # 'station_index': i, 
                'jc_loc': jc_loc, 
                'jb_loc': jb_loc, 
                'current_CH4': 0, 
                'height': station_height,
                'lon': station_lon,
                'lat': station_lat,
                'tracked_CH4': []
                })



@comin.register_callback(comin.EP_ATM_TIMELOOP_END) # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
def tracking_CH4_total():
    """tracking of CH4 Emissions"""
    global number_of_timesteps
    dtime = comin.descrdata_get_timesteplength(jg)
    datetime = comin.current_get_datetime() # This could maybe be useful for later, example for format: 2019-01-01T00:01:00.000
    number_of_timesteps += 1 # tracking number of steps, to in the end average over the correct time

    for i in range(len(local_monitoring_stations)):
        # Convert all of them to numpy arrays
        CH4_EMIS_np = np.asarray(CH4_EMIS)
        # CH4_TNO_np = np.asarray(CH4_TNO)
        # CH4_OMV_np = np.asarray(CH4_OMV)
        CH4_BG_np = np.asarray(CH4_BG)

        height = local_monitoring_stations[i]['height']
        jc_loc = local_monitoring_stations[i]['jc_loc']
        jb_loc = local_monitoring_stations[i]['jb_loc']

        # This is the main summation of all of the CH4 sources. Also not 100% sure if the indexing is correct
        local_monitoring_stations[i]['current_CH4'] += CH4_EMIS_np[jc_loc, height, jb_loc, 0, 0]
        # local_monitoring_stations[i]['current_CH4'] += CH4_TNO_np[jc_loc, height, jb_loc, 0, 0]
        # local_monitoring_stations[i]['current_CH4'] += CH4_OMV_np[jc_loc, height, jb_loc, 0, 0]
        local_monitoring_stations[i]['current_CH4'] += CH4_BG_np[jc_loc, height, jb_loc, 0, 0]

    elapsed_time = dtime * number_of_timesteps
    # Now this is where we log the averaged CH4, it is done by gathering the local monitoring stations and then storing them in an xarray, also writing them out then into an outputfile
    if (elapsed_time >= time_interval):
        local_data = None
        for i in range(len(local_monitoring_stations)):
            if(i==0):
                local_data = []
            avg_CH4 = local_monitoring_stations[i]['current_CH4'] / (dtime * number_of_timesteps)
            lon = local_monitoring_stations[i]['lon']
            lat = local_monitoring_stations[i]['lat']
            local_data.append( {'time_period': elapsed_time, 'avg_CH4': avg_CH4, 'datetime': datetime, 'lon': lon, 'lat': lat})
            local_monitoring_stations[i]['current_CH4'] = 0
        
        comm = MPI.COMM_WORLD
        newGroup = comm.group.Excl([123, 124, 125, 126, 127])
        newComm = comm.Create_group(newGroup)
        rank = newComm.Get_rank()
        gathered = newComm.gather(local_data, root=0)

        if newComm.Get_rank() == 0:
            gathered = [d for d in gathered if d is not None]
            flattened = [entry for sublist in gathered for entry in sublist]

            from collections import defaultdict

            station_data = defaultdict(list)
            time_label = None

            for entry in flattened:
                station_key = (entry["lon"], entry["lat"])
                station_data[station_key].append(entry["avg_CH4"].item())
                time_label = entry["datetime"]

            station_keys = sorted(station_data.keys())

            lons = [lon for lon, lat in station_keys]
            lats = [lat for lon, lat in station_keys]
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
                    "time": [pd.to_datetime(time_label)],
                },
            )

            ds["avg_CH4"].attrs["units"] = "ppbv"
            ds["avg_CH4"].attrs["long_name"] = "Average CH4 concentration"
            ds["lon"].attrs["units"] = "degrees_east"
            ds["lat"].attrs["units"] = "degrees_north"
            ds["time"].attrs["standard_name"] = "time"

            encoding = {
                "time": {
                    "units": "seconds since 2019-01-01 00:00:00",
                    "calendar": "proleptic_gregorian"
                }
            }

            output_file = "tracked_ch4.nc"

            global first_write_done
            if not first_write_done:
                ds.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")
                first_write_done = True
            else:
                existing_ds = xr.open_dataset(output_file)
                combined = xr.concat([existing_ds, ds], dim="time")
                existing_ds.close()
                combined.to_netcdf(output_file, mode="w", unlimited_dims=["time"], encoding=encoding, engine="netcdf4")
        number_of_timesteps = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def CH4_destructor():
    message("CH4_destructor called!", msgrank)
