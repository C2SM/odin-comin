#!/usr/bin/env python3
"""! @file tracker_anything.py
@brief Tracker plugin for sampling ICON variables using ComIn.

@section description_tracker_anything Description
Samples ICON variables at defined time points and averages over user-specified
start and end times for each sampling period. Supports monitoring stations,
satellite data, and CIF-based station/satellite sampling.

@section libraries_tracker_anything Libraries/Modules
- comin
- numpy
- mpi4py
- pandas
  - Access to pandas.to_datetime
- sys
- operator
- scipy
  - Access to scipy.spatial.KDTree
- sklearn
  - Access to sklearn.neighbors.BallTree
- netCDF4
  - Access to Dataset
- yaml
- datetime
- satellite (local module)
- monitoring_stations_final (local module)

@section author_tracker_anything Author(s)
- Created by Zeno Hug on 05/23/2025.

@section notes_tracker_anything Notes
Copyright (c) 2025 Empa. All rights reserved.
"""


# Imports
import comin
import numpy as np
from mpi4py import MPI
import pandas as pd
import sys
import operator
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from netCDF4 import Dataset
import yaml
import datetime as datetimelib

# Load config
with open('/capstor/scratch/cscs/zhug/Romania6km/plugin/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Ensure all factor values are float
for var, spec in config['dict_vars'].items():
    spec['factor'] = [float(f) for f in spec['factor']]

for var, spec in config['dict_vars_cif_sat'].items():
    spec['factor'] = [float(f) for f in spec['factor']]

for var, spec in config['dict_vars_cif_stations'].items():
    spec['factor'] = [float(f) for f in spec['factor']]

config['accepted_distance']= float(config['accepted_distance'])

# Global Constants from config
NUMBER_OF_NN = config['NUMBER_OF_NN']
time_interval_writeout = config['time_interval_writeout']
jg = config['jg']
msgrank = config['msgrank']
dict_vars = config['dict_vars']
dict_vars_cif_sat = config['dict_vars_cif_sat']
dict_vars_cif_stations = config['dict_vars_cif_stations']
do_monitoring_stations = config['do_monitoring_stations']
do_satellite_CH4 = config['do_satellite_CH4']
do_satellite_cif = config['do_satellite_cif']
do_stations_cif = config['do_stations_cif']
tropomi_filename = config['tropomi_filename']
cams_base_path = config['cams_base_path']
cams_params_file = config['cams_params_file']
path_to_input_nc = config['path_to_input_nc']
path_to_input_sat_cif = config['path_to_input_sat_cif']
path_to_input_stations_cif = config['path_to_input_stations_cif']
file_name_output_stations_cif = config['file_name_output_stations_cif']
file_name_output_sat_cif = config['file_name_output_sat_cif']
file_name_output = config['file_name_output']
file_name_output_sat_CH4 = config['file_name_output_sat_CH4']
accepted_distance = config['accepted_distance']

plugin_dir = config['plugin_dir']
if plugin_dir not in sys.path:
    sys.path.append(plugin_dir)

from satellite import *
from monitoring_stations import *

# Defining variables:
operations_dict = {
    "plus": operator.add,
    "minus": operator.sub,
}
cams_prev_data = None
cams_next_data = None

# Functions
def message(message_string, rank):
    """!Print a message from one PE only.

    @param message_string String to print.
    @param rank PE rank at which to print the message.
    """
    if (comin.parallel_get_host_mpi_rank() == rank):
        print(f"ComIn tracker_anything.py: {message_string}", file=sys.stderr)


def lonlat2xyz(lon, lat):
    """!Convert spherical lon/lat (radians) to unit-sphere Cartesian coordinates.

    @param lon Longitude(s) in radians (scalar or array).
    @param lat Latitude(s) in radians (scalar or array).
    @return Tuple (x, y, z) of coordinates.
    """
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def data_constructor():
    """!ComIn secondary constructor: request pointers to required ICON fields.

    Requests monitoring station variables, satellite CH4 fields, CIF variables,
    and station CIF variables based on configuration flags. Stores references
    for later use in the time loop.
    """
    global pres, pres_ifc, data, dict_vars, data, CH4_emis, CH4_bg, data_sat_cif, data_stations_cif
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

        
    if do_satellite_CH4:
        # Request to get the wanted variables (i.e. the EMIS and the BG. Also the pressure). We only want to read the data, not write
        CH4_emis = comin.var_get(entry_points, ("CH4_EMIS", jg), comin.COMIN_FLAG_READ)
        CH4_bg = comin.var_get(entry_points, ("CH4_BG", jg), comin.COMIN_FLAG_READ)
        pres_ifc = comin.var_get(entry_points, ("pres_ifc", jg), comin.COMIN_FLAG_READ)
    if do_satellite_CH4 or do_satellite_cif:
        pres = comin.var_get(entry_points, ("pres", jg), comin.COMIN_FLAG_READ)

    if do_satellite_cif:
        data_sat_cif = {}
        for variable, parameters in dict_vars_cif_sat.copy().items():
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
            data_sat_cif[variable] = local_data
    
    if do_stations_cif:
        data_stations_cif = {}
        for variable, parameters in dict_vars_cif_stations.copy().items():
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
            data_stations_cif[variable] = local_data
    
    message("data_constructor successful", msgrank)

@comin.register_callback(comin.EP_ATM_INIT_FINALIZE)
def stations_init():
    """!Initialize plugin: preprocess domain geometry and read station/satellite metadata.

    Sets up MPI communicator, KDTree for cell centers, ICON polygons for footprint
    calculations, and calls read-in functions for stations and satellites.  
    Also writes NetCDF headers for output files.
    """
    global number_of_timesteps, clon, hhl, decomp_domain, tree, num_levels # variables with domain info, and general information
    # All of the monitoring variables
    global cams_files_dict, data, data_satellite_to_do, data_satellite_done, data_monitoring_stations_to_do, data_monitoring_stations_done, dict_vars, data_sat_cif_to_do, data_sat_cif_done, data_stations_cif_to_do, data_stations_cif_done
    global dict_vars_cif_sat, dict_vars_cif_stations
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
    # exp_start = pd.to_datetime(simulation_interval.exp_start)
    # exp_stop = pd.to_datetime(simulation_interval.exp_stop)
    run_start = pd.to_datetime(simulation_interval.run_start)
    run_stop = pd.to_datetime(simulation_interval.run_stop)
    # All arrays are for domain 1 only
    # We read in all of the domain data we need. We read it in only once and then save it as a global variable
    domain = comin.descrdata_get_domain(jg)
    clon = np.asarray(domain.cells.clon)
    clat = np.asarray(domain.cells.clat)
    hhl = np.asarray(domain.cells.hhl)
    decomp_domain = np.asarray(domain.cells.decomp_domain)
    vertex_idx = np.asarray(domain.cells.vertex_idx)
    vertex_blk = np.asarray(domain.cells.vertex_blk)
    vlon = np.asarray(domain.verts.vlon)
    vlat = np.asarray(domain.verts.vlat)
    num_levels = domain.nlev
    icon_polygons = []

    if do_satellite_CH4:
        for i in range(clon.shape[0]):
            for j in range(clon.shape[1]):
                coords = []
                for k in range(3):
                        idx = vertex_idx[i, j, k] - 1 # - 1 because fortran indexing starts at 1 and not at 0
                        blk = vertex_blk[i, j, k] - 1
                        # if the indexes are -1, this is an invalid cell, which does not have a positive area. But we still need it in the icon polygons, as we want the same order as clon with all cells of clon, if not we can't compare them anymore
                        # Also, pure observation showed that if one corner is invalid for a cell, then all of them are. But, this could be a source of error at some point
                        # if idx == -1 and blk == -1:
                        #     message(f"Please investigate this cell. It could be interesting: i: {i}, j: {j}, k: {k}, idx: {idx}, blk: {blk}", msgrank)
                            # continue
                        lon = vlon[idx, blk]
                        lat = vlat[idx, blk]
                        coords.append((lon, lat))
                icon_polygons.append(Polygon(coords))
    
    # Convert the longitude latitude data to xyz data for the KDTree
    xyz = np.c_[lonlat2xyz(clon.ravel(),clat.ravel())]
    tree = KDTree(xyz) # Create the KDTree, composed of the xyz data

    # Get all of the monitoring stations and save all of the relevant data
    if(do_monitoring_stations):
        message("Now reading in and locating the correct indices for the monitoring stations", msgrank)
        data_monitoring_stations_to_do, data_monitoring_stations_done = read_in_points(comm, tree, decomp_domain, clon, hhl, NUMBER_OF_NN, path_to_input_nc, run_start, run_stop, data, accepted_distance)
        write_header_points(comm, file_name_output, dict_vars)

    if(do_satellite_CH4):
        message("Now reading in and locating the correct indices for the satellite data", msgrank)
        data_satellite_to_do, data_satellite_done, cams_files_dict = read_in_satellite_data_CH4(comm, tree, decomp_domain, clon, run_start, run_stop, tropomi_filename, cams_base_path, cams_params_file, accepted_distance, icon_polygons)
        write_header_sat(comm, file_name_output_sat_CH4)

    if(do_satellite_cif):
        message("Now reading in and locating the correct indices for the cif satellite data", msgrank)
        data_sat_cif_to_do, data_sat_cif_done = read_in_satellite_data_cif(comm, tree, decomp_domain, clon, run_start, run_stop, path_to_input_sat_cif, NUMBER_OF_NN, accepted_distance, num_levels)
        write_header_sat_cif(comm, file_name_output_sat_cif, num_levels)
    if(do_stations_cif):
        message("Now reading in and locating the correct indices for the cif station data", msgrank)
        data_stations_cif_to_do, data_stations_cif_done = read_in_points_cif(comm, tree, decomp_domain, clon, hhl, NUMBER_OF_NN,path_to_input_stations_cif, run_start, run_stop, dict_vars_cif_stations, accepted_distance)
        write_header_points_cif(comm, file_name_output_stations_cif)
    message("Done reading in all data", msgrank)


@comin.register_callback(comin.EP_ATM_TIMELOOP_START)
def read_in_cams():
    """!Read CAMS data if current model time is 0, 6, 12, or 18 UTC.

    Opens the pair of CAMS datasets bracketing the current model time using
    ``update_cams``.
    """
    # The cams data which will get changed
    global cams_prev_data, cams_next_data

    # get the datetime from comin
    datetime = comin.current_get_datetime()

    # Read in the CAMS data if it is the right time
    if do_satellite_CH4 and (pd.to_datetime(datetime).time() == datetimelib.time(0,0) or pd.to_datetime(datetime).time() == datetimelib.time(6,0) or pd.to_datetime(datetime).time() == datetimelib.time(12,0) or pd.to_datetime(datetime).time() == datetimelib.time(18,0)):
        cams_prev_data, cams_next_data = update_cams(datetime, cams_files_dict, cams_prev_data, cams_next_data)


@comin.register_callback(comin.EP_ATM_TIMELOOP_END)
def tracking():
    """!Main tracking callback: process observations and write results periodically.

    Converts ComIn variable pointers to NumPy arrays, updates monitoring station,
    satellite CH4, satellite CIF, and station CIF measurements. Writes results to
    NetCDF when the configured output interval is reached.
    """
    # general info
    global number_of_timesteps, data, dict_vars, operations_dict, dict_vars_cif_stations, dict_vars_cif_sat, data_sat_cif, data_stations_cif
    # satellite and point data
    global data_satellite_to_do, data_satellite_done, data_monitoring_stations_done, data_monitoring_stations_to_do, data_sat_cif_to_do, data_sat_cif_done, data_stations_cif_to_do, data_stations_cif_done

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

    if do_satellite_CH4:
        CH4_EMIS_np = np.asarray(CH4_emis)
        CH4_BG_np = np.asarray(CH4_bg)
        pres_np = np.asarray(pres)
        pres_ifc_np = np.asarray(pres_ifc)

    if do_satellite_cif:
        pres_np = np.asarray(pres)
        data_np_sat_cif = {}
        for variable, list in data_sat_cif.items():
            local_data = []
            for item in list:
                local_data.append(np.asarray(item))
            data_np_sat_cif[variable] = local_data

    if do_stations_cif:
        data_np_stations_cif = {}
        for variable, list in data_stations_cif.items():
            local_data = []
            for item in list:
                local_data.append(np.asarray(item))
            data_np_stations_cif[variable] = local_data

    # Monitoring of the variables for the different problems
    if do_monitoring_stations:
        tracking_points(datetime, data_monitoring_stations_to_do, data_monitoring_stations_done, data_np, dict_vars, operations_dict)
    if do_satellite_CH4:
        tracking_CH4_satellite(datetime, CH4_EMIS_np, CH4_BG_np, pres_ifc_np, pres_np, data_satellite_to_do, data_satellite_done, cams_prev_data, cams_next_data)
    if do_satellite_cif:
        tracking_satellite_pressures(datetime, data_sat_cif_to_do, data_sat_cif_done, data_np_sat_cif, dict_vars_cif_sat, operations_dict, pres_np, num_levels)
    if do_stations_cif:
        tracking_points_cif(datetime, data_stations_cif_to_do, data_stations_cif_done, data_np_stations_cif, dict_vars_cif_stations, operations_dict)
        

    ## Writeout
    elapsed_time = dtime * number_of_timesteps # Time that has elapsed since last writeout (or since start if there was no writeout yet)
    if (elapsed_time >= time_interval_writeout): # If we are above the time intervall writeout, we want to write out
        if do_monitoring_stations:
            write_points(comm, data_monitoring_stations_done, dict_vars, file_name_output)
        if do_satellite_CH4:
            write_satellite_CH4(comm, data_satellite_done, file_name_output_sat_CH4)
        if do_satellite_cif:
            write_satellite_cif(comm, data_sat_cif_done, file_name_output_sat_cif)
        if do_stations_cif:
            write_points_cif(comm, data_stations_cif_done, file_name_output_stations_cif)

        # Reset data
        number_of_timesteps = 0
        


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    """!Destructor callback: perform final cleanup actions.

    Currently logs that the destructor has been called.
    """
    message("destructor called!", msgrank)
