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
# from shapely.geometry import Polygon
# from shapely.strtree import STRtree
import datetime as datetimelib
import os

# Manually specify the plugin directory — adjust if your path changes
plugin_dir = "/capstor/scratch/cscs/zhug/Romania6km/python-zeno"
if plugin_dir not in sys.path:
    sys.path.append(plugin_dir)

from monitoring_stations import *
from flight import *
from satellite import *

## Constants:
NUMBER_OF_NN = 4 # Number of nearest neighbour cells over which should be interpolated
N_COMPUTE_PES = 123  # number of compute PEs
jg = 1 # we do compututations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
days_of_flights = [[2019, 10, 7], [2019, 10, 8]] # Up until now manually put in the dates of the flights
start_model = datetimelib.datetime(2019, 1, 1)
do_monitoring_stations = True
do_satellite = True
do_flights = True
# tropomi_filename = "TROPOMI_SRON_prs_flipped_20190101_20191231.nc"
tropomi_filename = "TROPOMI_SRON_20190101_20191231.nc"
cams_base_path = "/capstor/scratch/cscs/zhug/Romania6km/input/CAMS/LBC/"
cams_params_file = "cams73_v22r2_ch4_conc_surface_inst_201910.nc"
days_of_flights_datetime = []
for entry in days_of_flights:
    days_of_flights_datetime.append(datetimelib.date(entry[0], entry[1], entry[2]))
time_interval_writeout = 900 # variable saying how often you want to writeout the results, in seconds

## Defining variables:
first_write_done_monitoring = False # Help to know if output file already exists

cams_prev_data = None
cams_next_data = None

def message(message_string, rank):
    """Short helper function to print a message on one PE"""
    if (comin.parallel_get_host_mpi_rank() == rank):
        print(f"ComIn point_source.py: {message_string}", file=sys.stderr)

def lonlat2xyz(lon, lat):
    """Short helper function for calculating xyz coordinates from longitues and latitudes"""
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def data_constructor():
    """Constructor: Get pointers to data"""
    global CH4_emis, CH4_bg, pres, pres_ifc
    entry_points = [comin.EP_ATM_TIMELOOP_END] # TIMELOOP END is atm selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
    
    # Request to get the wanted variables (i.e. the EMIS and the BG. Also the pressure). We only want to read the data, not write
    CH4_emis = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
    CH4_bg = comin.var_get(entry_points, ("CH4_BG", jg), comin.COMIN_FLAG_READ)
    pres = comin.var_get(entry_points, ("pres", jg), comin.COMIN_FLAG_READ)
    pres_ifc = comin.var_get(entry_points, ("pres_ifc", jg), comin.COMIN_FLAG_READ)

    message("data_constructor successful", msgrank)

@comin.register_callback(comin.EP_ATM_INIT_FINALIZE)
def stations_init():
    global number_of_timesteps, clon, hhl, decomp_domain, tree # variables with domain info, and general information
    # All of the monitoring variables
    global cams_files_dict
    # MPI variables
    global comm, rank
    global data_monitoring_stations, data_flight_to_do, data_flight_done, data_satellite_to_do, data_satellite_done

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
    
    datetime = comin.current_get_datetime()
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

    # Get all of the monitoring stations and save all of the relevant data
    if(do_monitoring_stations):
        data_monitoring_stations = read_in_monitoring_stations(datetime, comm, tree, decomp_domain, clon, hhl, NUMBER_OF_NN)

    if(do_flights):
        data_flight_to_do, data_flight_done = initialize_empty()

    if(do_satellite):
        data_satellite_to_do, data_satellite_done, cams_files_dict = read_in_satellite_data(datetime, comm, tree, decomp_domain, clon, start_model, tropomi_filename, cams_base_path, cams_params_file)

@comin.register_callback(comin.EP_ATM_TIMELOOP_START)
def input_data_on_the_fly():
    """reading in files"""
    # All of the singlepoints variables
    global data_flight_to_do, data_flight_done

    global cams_prev_data, cams_next_data

    # get the datetime from comin
    datetime = comin.current_get_datetime()

    # Currently we read in at midnight for the following day
    if(do_flights and pd.to_datetime(datetime).time() == datetimelib.time(0,0) and datetime in days_of_flights_datetime):
       data_flight_to_do, data_flight_done = read_in_flight_data(datetime, comm, tree, decomp_domain, clon, hhl, NUMBER_OF_NN)

    # Read in the CAMS data
    if do_satellite and (pd.to_datetime(datetime).time() == datetimelib.time(0,0) or pd.to_datetime(datetime).time() == datetimelib.time(6,0) or pd.to_datetime(datetime).time() == datetimelib.time(12,0) or pd.to_datetime(datetime).time() == datetimelib.time(18,0)):
        cams_prev_data, cams_next_data = update_cams(datetime, cams_files_dict, cams_prev_data, cams_next_data)

@comin.register_callback(comin.EP_ATM_TIMELOOP_END) # TIMELOOP END is atm randomly selected, as it's just once every iteration, maybe it makes sense to have a different Entry Point
def tracking_CH4_total():
    """tracking of CH4 Emissions"""
    # general info
    global number_of_timesteps, first_write_done_monitoring
    # stationary monitoring
    global data_monitoring_stations, data_flight_to_do, data_flight_done, data_satellite_to_do, data_satellite_done

    dtime = comin.descrdata_get_timesteplength(jg) # size of every timestep 
    datetime = comin.current_get_datetime() # get datetime info. example for format: 2019-01-01T00:01:00.000
    number_of_timesteps += 1 # tracking number of steps

    # Convert all of them to numpy arrays
    CH4_EMIS_np = np.asarray(CH4_emis)
    CH4_BG_np = np.asarray(CH4_bg)
    pres_np = np.asarray(pres)
    pres_ifc_np = np.asarray(pres_ifc)

    ## Monitoring of the CH4 for the different problems
    if do_monitoring_stations:
        tracking_CH4_monitoring(datetime, CH4_EMIS_np, CH4_BG_np, data_monitoring_stations)
    if do_flights:
        tracking_CH4_flight(datetime, CH4_EMIS_np, CH4_BG_np, data_flight_to_do, data_flight_done)
    if do_satellite:
        tracking_CH4_satellite(datetime, comm, CH4_EMIS_np, CH4_BG_np, pres_ifc_np, pres_np, data_satellite_to_do, data_satellite_done, cams_prev_data, cams_next_data, cams_files_dict)
        

    ## Writeout
    elapsed_time = dtime * number_of_timesteps # Time that has elapsed since last writeout (or since start if there was no writeout yet)
    if (elapsed_time >= time_interval_writeout): # If we are above the time intervall writeout defined on the very top, we want to write out
        if do_monitoring_stations:
            first_write_done_monitoring = write_monitoring_stations(datetime, comm, data_monitoring_stations, first_write_done_monitoring) # Write out the monitoring stations
        if do_flights:
            write_singlepoints(datetime, comm, data_flight_done) # Write out the flight data that is done
        if do_satellite:
            write_satellite(datetime, comm, data_satellite_done) # Write out the satellite data that is done

        # Reset data
        number_of_timesteps = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def CH4_destructor():
    message("CH4_destructor called!", msgrank)
