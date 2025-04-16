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
import sys
from scipy.spatial import KDTree

jg = 1 # we do compututations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
monitoring_stations = [[26.0, 46.0], [23.0, 47.0]] # first coordinate is longitude, second coordinate is latitude
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
    global CH4_EMIS, CH4_TNO, CH4_OMV, CH4_BG
    entry_points = [comin.EP_ATM_TIMELOOP_END] # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
    
    CH4_EMIS = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
    CH4_TNO = comin.var_get(entry_points, ("CH4_TNO", jg), comin.COMIN_FLAG_READ)
    CH4_OMV = comin.var_get(entry_points, ("CH4_OMV", jg), comin.COMIN_FLAG_READ)
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
                'tracked_CH4': []
                })



@comin.register_callback(comin.EP_ATM_TIMELOOP_END) # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
def tracking_CH4_total():
    """tracking of CH4 Emissions"""
    global number_of_timesteps
    dtime = comin.descrdata_get_timesteplength(jg)
    datetime = comin.current_get_datetime() # This could maybe be useful for later, example for format: 2019-01-01T00:01:00.000

    for i in range(len(local_monitoring_stations)):
        # Convert all of them to numpy arrays
        CH4_EMIS_np = np.asarray(CH4_EMIS)
        CH4_TNO_np = np.asarray(CH4_TNO)
        CH4_OMV_np = np.asarray(CH4_OMV)
        CH4_BG_np = np.asarray(CH4_BG)

        # This is the main summation of all of the CH4 sources. Maybe some of them are counted twice, will have to look at again, but very simple to take out a source. Also not 100% sure if the indexing is correct
        local_monitoring_stations[i]['current_CH4'] += CH4_EMIS_np[local_monitoring_stations[i]['jc_loc'],0,local_monitoring_stations[i]['jb_loc'],0,0]
        local_monitoring_stations[i]['current_CH4'] += CH4_TNO_np[local_monitoring_stations[i]['jc_loc'],0,local_monitoring_stations[i]['jb_loc'],0,0]
        local_monitoring_stations[i]['current_CH4'] += CH4_OMV_np[local_monitoring_stations[i]['jc_loc'],0,local_monitoring_stations[i]['jb_loc'],0,0]
        local_monitoring_stations[i]['current_CH4'] += CH4_BG_np[local_monitoring_stations[i]['jc_loc'],0,local_monitoring_stations[i]['jb_loc'],0,0]
        # tracking number of steps, to in the end average over the correct time
        if (i == 0):
            number_of_timesteps += 1


    # Now this is where we log the averaged CH4, currently it is done by just storing it in the local monitoring stations variable, will do differently in future
    if (dtime * number_of_timesteps >= time_interval):
        for i in range(len(local_monitoring_stations)):
            # avg_CH4 = local_monitoring_stations[i]['current_CH4'] / (dtime * local_monitoring_stations[i]['number_of_timesteps'])
            # local_monitoring_stations[i]['tracked_CH4'].append({'time': dtime * local_monitoring_stations[i]['number_of_timesteps'], 'avg_CH4': avg_CH4})
            avg_CH4 = local_monitoring_stations[i]['current_CH4'] / (dtime * number_of_timesteps)
            local_monitoring_stations[i]['tracked_CH4'].append({'time': dtime * number_of_timesteps, 'avg_CH4': avg_CH4, 'datetime': datetime})
            local_monitoring_stations[i]['current_CH4'] = 0
            # print(f"ComIn point_source.py: {local_monitoring_stations[i]['tracked_CH4']}, called by process {comin.parallel_get_host_mpi_rank()}, LOL EY SO TOLL", file=sys.stderr)
        
        number_of_timesteps = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def CH4_destructor():
    message("CH4_destructor called!", msgrank)
