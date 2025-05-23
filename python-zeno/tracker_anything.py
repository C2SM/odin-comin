#!/usr/bin/env python3
"""! @brief Plugin for monitoring ICON variables"""


##
# @mainpage Plugin
#
# @section description_main Description
# Plugin for monitoring ICON variables on different timepoints and averaging over a starting time of the measurement and an ending time of the measurement
# Also able to track CH4 data from satellite data
#
# @section notes_main Notes
#
# Copyright (c) 2025 Empa. All rights reserved

##
# @file tracker_anything.py
#
# @brief Tracker for user defined variables using ICON ComIn
#
# @section description_tracker_anything Description
# Tracker for user defined variables using ICON ComIn
#
# @section libraries_main Libraries/Modules
# - comin library
# - numpy library
# - mpi4py library
# - pandas library
#   - Access to pandas.to_datetime
# - sys library
# - operator library
# - scipy library
#   - Access to scipy.spatial.KDTree
# - netCDF4 library
#   - Access to Dataset
# - datetime library
# - satellite module (local)
# - monitoring_stations_final module (local)
#
# @section author_tracker_anything Author(s)
# - Created by Zeno Hug on 05/23/2025.
#
# Copyright (c) 2025 Empa.  All rights reserved.

# Imports
import comin
import numpy as np
from mpi4py import MPI
import pandas as pd
import sys
import operator
from scipy.spatial import KDTree
from netCDF4 import Dataset
import datetime as datetimelib

# Manually specify the plugin directory
plugin_dir = "/capstor/scratch/cscs/zhug/Romania6km/python-zeno"
if plugin_dir not in sys.path:
    sys.path.append(plugin_dir)

from satellite import *
from monitoring_stations_final import *

## Global Constants:
NUMBER_OF_NN = 4 # Number of nearest neighbour cells over which should be interpolated
time_interval_writeout = 3600 # variable saying how often you want to writeout the results, in seconds
jg = 1 # we do computations only on domain 1, as in our case our grid only has one domain
msgrank = 0 # Rank that prints messages
dict_vars = {   'CH4': {   'var_names': ['CH4_EMIS', 'CH4_BG'],
                            'signs': ['plus'],
                            'factor': [1e9, 1], 
                            'unit': 'ppb', 
                            'long_name': 'CH4 concentration'
                        }, 
                'Temp': {   'var_names': ['temp'],
                            'signs': [],
                            'factor': [1], 
                            'unit': 'Kelvin', 
                            'long_name': 'Temperature'
                        }
            }
do_monitoring_stations = True
do_satellite = True
tropomi_filename = "TROPOMI_SRON_prs_flipped_20190101_20191231.nc"
cams_base_path = "/capstor/scratch/cscs/zhug/Romania6km/input/CAMS/LBC/"
cams_params_file = "/capstor/scratch/cscs/zhug/Romania6km/input/CAMS/cams73_v22r2_ch4_conc_surface_inst_201910.nc"
path_to_input_nc = "/capstor/scratch/cscs/zhug/Romania6km/input/full_stations/input.nc"
file_name_output = "output.nc"
file_name_output_sat = "output_sat.nc"

## Defining variables:
operations_dict = {
    "plus": operator.add,
    "minus": operator.sub,
}
cams_prev_data = None
cams_next_data = None

# Functions
def message(message_string, rank):
    """! Short helper function to print a message on one PE
    @param message_string   string, the message which you want to print
    @param rank             The rank at which the message should be printed
    """
    if (comin.parallel_get_host_mpi_rank() == rank):
        print(f"ComIn tracker_anything.py: {message_string}", file=sys.stderr)


def lonlat2xyz(lon, lat):
    """! Short helper function for calculating xyz coordinates from longitues and latitudes
    @param lon   An array or single value of longitudes to convert to xyz coordinates
    @param lat   An array or single value of latitudes to convert to xyz coordinates
    @return  converted xyz values as a tuple
    """
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def data_constructor():
    """! Constructor: Get pointers to data
    """
    global pres, pres_ifc, data, dict_vars, data, CH4_emis, CH4_bg
    entry_points = [comin.EP_ATM_TIMELOOP_END]
    
    if do_monitoring_stations:
        data = {}
        for variable, parameters in dict_vars.copy().items():
            message(f"Now working on variable {variable}", msgrank)
            # Check if there is the right amount of entries. Could also do more checks here
            if len(parameters['signs']) != len(parameters['var_names']) - 1 or len(parameters['factor']) != len(parameters['var_names']):
                message(f"   Please provide the right amount of variables, signs and factors, will skip {variable}", msgrank)
                del dict_vars[variable]
            local_data = []
            for var_name in parameters['var_names']:
                message(f"  Now trying to access {var_name} in ComIn. If it crashes after this you misspelled this or this variable doesn't exist in ICON. Please double check", msgrank)
                # Request to get the wanted variables. We only want to read the data, not write
                local_data.append(comin.var_get(entry_points, (var_name, jg), comin.COMIN_FLAG_READ))
            data[variable] = local_data

        
    if do_satellite:
        # Request to get the wanted variables (i.e. the EMIS and the BG. Also the pressure). We only want to read the data, not write
        CH4_emis = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
        CH4_bg = comin.var_get(entry_points, ("CH4_BG", jg), comin.COMIN_FLAG_READ)
        pres = comin.var_get(entry_points, ("pres", jg), comin.COMIN_FLAG_READ)
        pres_ifc = comin.var_get(entry_points, ("pres_ifc", jg), comin.COMIN_FLAG_READ)

    message("data_constructor successful", msgrank)

@comin.register_callback(comin.EP_ATM_INIT_FINALIZE)
def stations_init():
    """! Initialization: Get data and preprocess all of the data
    """
    global number_of_timesteps, clon, hhl, decomp_domain, tree # variables with domain info, and general information
    # All of the monitoring variables
    global cams_files_dict, data, data_satellite_to_do, data_satellite_done, data_monitoring_stations_to_do, data_monitoring_stations_done, dict_vars
    # MPI variables
    global comm

    # Ask ComIn for the raw MPI_Comm that ICON itself is using
    comm_handle = comin.parallel_get_host_mpi_comm()
    
    # Convert that handle into a regular mpi4py communicator
    comm = MPI.Comm.f2py(comm_handle)

    number_of_timesteps = 0 # number of timesteps, as we are initializing we set it to 0
    
    datetime = comin.current_get_datetime()
    datetime_np = pd.to_datetime(datetime)

    simulation_interval = comin.descrdata_get_simulation_interval()
    # exp_start = simulation_interval.exp_start 
    # exp_stop = simulation_interval.exp_stop
    # exp_start = pd.to_datetime(exp_start)
    # exp_stop = pd.to_datetime(exp_stop)
    run_start = pd.to_datetime(simulation_interval.run_start)
    run_stop = pd.to_datetime(simulation_interval.run_stop)
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
        data_monitoring_stations_to_do, data_monitoring_stations_done = read_in_points(comm, tree, decomp_domain, clon, hhl, NUMBER_OF_NN, path_to_input_nc, run_start, run_stop, data)
        # data_monitoring_stations_to_do, data_monitoring_stations_done = read_in_points(comm, tree, decomp_domain, clon, hhl, NUMBER_OF_NN, path_to_input_nc, exp_start, exp_stop, data)
    if(do_satellite):
        data_satellite_to_do, data_satellite_done, cams_files_dict = read_in_satellite_data(comm, tree, decomp_domain, clon, run_start, run_stop, tropomi_filename, cams_base_path, cams_params_file)
        # data_satellite_to_do, data_satellite_done, cams_files_dict = read_in_satellite_data(comm, tree, decomp_domain, clon, exp_start, exp_stop, tropomi_filename, cams_base_path, cams_params_file)
    if(comm.Get_rank() == 0):

        # Append all of the variables we want to measure to the output nc file. This assumes that the rest of the header of the nc file is already written
        if do_monitoring_stations:
            for variable, parameters in dict_vars.items():
                ncfile = Dataset(file_name_output, 'a')
                temp_var = ncfile.createVariable(variable, 'f8', ('obs',))
                temp_var.units = parameters['unit']
                temp_var.long_name = parameters['long_name']
                ncfile.close()

        # Create the header for the satellite data, it is the in the same format as the tropomi data
        if do_satellite:
            ncfile_sat = Dataset(file_name_output_sat, 'w', format='NETCDF4')
            index = ncfile_sat.createDimension('index', None)
            date = ncfile_sat.createVariable('date', 'u8',('index',) )
            date.units = "milliseconds since 2019-01-01 11:14:35.629000" 
            date.calendar = "proleptic_gregorian"
            lon = ncfile_sat.createVariable('lon', 'f8',('index',) )
            lat = ncfile_sat.createVariable('lat', 'f8',('index',) )
            ch4 = ncfile_sat.createVariable('CH4', 'f8',('index',) )
            ncfile_sat.close()


@comin.register_callback(comin.EP_ATM_TIMELOOP_START)
def read_in_cams():
    """! Read in CAMS files on the fly if it is 0, 6, 12 or 18 o'clock
    """
    # The cams data which will get changed
    global cams_prev_data, cams_next_data

    # get the datetime from comin
    datetime = comin.current_get_datetime()

    # Read in the CAMS data if it is the right time
    if do_satellite and (pd.to_datetime(datetime).time() == datetimelib.time(0,0) or pd.to_datetime(datetime).time() == datetimelib.time(6,0) or pd.to_datetime(datetime).time() == datetimelib.time(12,0) or pd.to_datetime(datetime).time() == datetimelib.time(18,0)):
        cams_prev_data, cams_next_data = update_cams(datetime, cams_files_dict, cams_prev_data, cams_next_data)


@comin.register_callback(comin.EP_ATM_TIMELOOP_END)
def tracking():
    """! Tracking: Track the data and writeout the done data in intervals
    """
    # general info
    global number_of_timesteps, data, dict_vars, operations_dict
    # satellite and point data
    global data_satellite_to_do, data_satellite_done, data_monitoring_stations_done, data_monitoring_stations_to_do

    dtime = comin.descrdata_get_timesteplength(jg) # size of every timestep 
    datetime = comin.current_get_datetime() # get datetime info. example for format: 2019-01-01T00:01:00.000
    number_of_timesteps += 1 # tracking number of steps

    # Convert all of the data to numpy arrays
    if do_monitoring_stations:
        data_np = {}
        for variable, list in data.items():
            local_data = []
            for item in list:
                local_data.append(np.asarray(item))
            data_np[variable] = local_data

    if do_satellite:
        CH4_EMIS_np = np.asarray(CH4_emis)
        CH4_BG_np = np.asarray(CH4_bg)
        pres_np = np.asarray(pres)
        pres_ifc_np = np.asarray(pres_ifc)

    # Monitoring of the variables for the different problems
    if do_monitoring_stations:
        tracking_points(datetime, data_monitoring_stations_to_do, data_monitoring_stations_done, data_np, dict_vars, operations_dict)
    if do_satellite:
        tracking_CH4_satellite(datetime, CH4_EMIS_np, CH4_BG_np, pres_ifc_np, pres_np, data_satellite_to_do, data_satellite_done, cams_prev_data, cams_next_data)
        

    ## Writeout
    elapsed_time = dtime * number_of_timesteps # Time that has elapsed since last writeout (or since start if there was no writeout yet)
    if (elapsed_time >= time_interval_writeout): # If we are above the time intervall writeout, we want to write out
        if do_monitoring_stations:
            write_points(comm, data_monitoring_stations_done, dict_vars, file_name_output)
        if do_satellite:
            write_satellite(comm, data_satellite_done, file_name_output_sat)

        # Reset data
        number_of_timesteps = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    """! Destructor
    """
    message("destructor called!", msgrank)
