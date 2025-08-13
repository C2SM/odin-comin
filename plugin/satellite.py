"""! @file satellite.py

@brief Tracking data for comparison with satellite observations.

@section description_satellite Description
Tracking data to compare to satellite data.

@section libraries_satellite Libraries/Modules
- numpy
- xarray
- netCDF4
  - Access to Dataset
- pandas
  - Access to pandas.to_datetime and pandas.read_csv
- os
- sys
- datetime
- scipy
  - Access to scipy.spatial.KDTree
- shapely
  - Access to shapely.Polygon and shapely.STRtree
- get_int_coefs (local module)

@section author_satellite Author(s)
- Created by Zeno Hug on 08/11/2025.

Copyright (c) 2025 Empa. All rights reserved.
"""



import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import KDTree
from shapely import Polygon, STRtree
import datetime as datetimelib
from get_int_coefs import *
import sys
import os

## Constants (molar masses; dry air and CH4)
Mda, MCH4 = 28.964, 16.04

class SatelliteDataToDoGeneral:
    """!Container for pending satellite collocations (generic parameter).

    Attributes:
        lon (np.ndarray): Longitudes of satellite ground pixels.
        lat (np.ndarray): Latitudes of satellite ground pixels.
        timestep (np.ndarray): Observation datetimes.
        jc_loc (np.ndarray): JC indices of contributing ICON cells.
        jb_loc (np.ndarray): JB indices of contributing ICON cells.
        parameter (np.ndarray): Parameter name per observation.
        weights (np.ndarray): Horizontal interpolation weights per obs.
    """
    def __init__(self, lon, lat, timestep, jc_loc, jb_loc, parameter, weights):
        """!Initialize a SatelliteDataToDoGeneral instance.

        @param lon Longitudes.
        @param lat Latitudes.
        @param timestep Observation datetimes.
        @param jc_loc JC indices.
        @param jb_loc JB indices.
        @param parameter Parameter names.
        @param weights Horizontal interpolation weights.
        """
        self.lon = lon
        self.lat = lat
        self.timestep = timestep
        self.jc_loc = jc_loc
        self.jb_loc = jb_loc
        self.parameter = parameter
        self.weights = weights

    def filter_ready(self, current_time):
        """!Return a mask of observations ready to be processed.

        @param current_time Current model time.
        @return Boolean mask where ``timestep <= current_time``.
        """
        mask = self.timestep <= np.datetime64(current_time)
        return mask

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes with matching length.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])

class SatelliteDataToDoCH4:
    """!Container for pending CH4 satellite collocations with retrieval metadata.

    Attributes:
        lon, lat (np.ndarray): Ground pixel positions.
        timestep (np.ndarray): Observation datetimes.
        jc_loc, jb_loc (np.ndarray): ICON cell indices per obs.
        weights (np.ndarray): Horizontal interpolation weights.
        qa0, ak, pw, pavg0 (np.ndarray): Retrieval vectors/matrices.
        cams_index (np.ndarray): CAMS nearest-index per obs.
        frac_cams (np.ndarray): Fraction between 6-hour CAMS steps [0,1].
        hyam, hybm, hyai, hybi (np.ndarray): CAMS hybrid coeffs.
        covered_areas (np.ndarray): Fractional coverage over ICON cells.
    """
    def __init__(self, lon, lat, timestep, jc_loc, jb_loc, weights, qa0, ak, pw, pavg0, cams_index, frac_cams, hyam, hybm, hyai, hybi, covered_areas):
        """!Initialize a SatelliteDataToDoCH4 instance.

        @param lon Longitudes.
        @param lat Latitudes.
        @param timestep Observation datetimes.
        @param jc_loc JC indices.
        @param jb_loc JB indices.
        @param weights Horizontal interpolation weights.
        @param qa0 Retrieval prior (per level).
        @param ak Averaging kernel (per level).
        @param pw Pressure weighting function (per level).
        @param pavg0 Pressure grid of retrieval (interfaces or midpoints).
        @param cams_index CAMS nearest grid index per obs.
        @param frac_cams Linear interpolation fraction between CAMS steps.
        @param hyam, hybm, hyai, hybi CAMS hybrid coefficients.
        @param covered_areas Fractional area of satellite pixel covering ICON cells.
        """
        self.lon = lon
        self.lat = lat
        self.timestep = timestep
        self.jc_loc = jc_loc
        self.jb_loc = jb_loc
        self.weights = weights

        self.qa0 = qa0
        self.ak = ak
        self.pw = pw
        self.pavg0 = pavg0
        self.cams_index = cams_index
        self.frac_cams = frac_cams
        self.hyam = hyam
        self.hybm = hybm
        self.hyai = hyai
        self.hybi = hybi
        self.covered_areas = covered_areas

    def filter_ready(self, current_time):
        """!Return a mask of observations ready to be processed.

        @param current_time Current model time.
        @return Boolean mask where ``timestep <= current_time``.
        """
        mask = self.timestep <= np.datetime64(current_time)
        return mask

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes with matching length.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])

class SatelliteDataDoneGeneral:
    """!Container for completed satellite column results (generic parameter).

    Attributes:
        lon, lat (np.ndarray): Positions of completed observations.
        timestep (np.ndarray): Observation datetimes.
        parameter (np.ndarray): Parameter names.
        measurement (np.ndarray): Column vectors per obs (shape: N x level).
        counter (int): Number of filled entries.
    """
    def __init__(self, lon, lat, timestep, parameter, measurement, counter):
        """!Initialize a SatelliteDataDoneGeneral instance.

        @param lon Longitudes.
        @param lat Latitudes.
        @param timestep Observation datetimes.
        @param parameter Parameter names.
        @param measurement Column values per level.
        @param counter Number of filled entries.
        """
        self.lon = lon
        self.lat = lat
        self.timestep = timestep
        self.parameter = parameter
        self.measurement = measurement
        self.counter = counter

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes with matching length.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])

class SatelliteDataDoneCH4:
    """!Container for completed CH4 satellite comparisons (ready to write out).

    Attributes:
        lon, lat (np.ndarray): Positions of completed observations.
        timestep (np.ndarray): Observation datetimes.
        CH4 (np.ndarray): Retrieved/compared total columns (per obs).
        counter (int): Number of filled entries.
        covered_areas (np.ndarray): Fractional coverage written alongside.
    """
    def __init__(self, lon, lat, timestep, CH4, counter, covered_areas):
        """!Initialize a SatelliteDataDoneCH4 instance.

        @param lon Longitudes.
        @param lat Latitudes.
        @param timestep Observation datetimes.
        @param CH4 CH4 column values.
        @param counter Number of filled entries.
        @param covered_areas Fractional coverage per obs.
        """
        self.lon = lon
        self.lat = lat
        self.timestep = timestep
        self.CH4 = CH4
        self.counter = counter
        self.covered_areas = covered_areas

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes with matching length.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])

def pad_list_of_lists_float(list_of_lists, pad_value=0.0):
    """!Pad a list of lists to a 2D float NumPy array with a given pad value.

    @param list_of_lists List of lists to pad/stack.
    @param pad_value Value used to pad shorter rows.
    @return 2D NumPy array (dtype=float64).
    """
    if not list_of_lists:
        return np.array([], dtype=np.float64).reshape(0, 0)

    max_len = max(len(inner) for inner in list_of_lists)
    padded_lists = [list(inner) + [pad_value] * (max_len - len(inner)) for inner in list_of_lists]
    return np.array(padded_lists, dtype=np.float64)

def pad_list_of_lists_int(list_of_lists, pad_value=0):
    """!Pad a list of lists to a 2D int NumPy array with a given pad value.

    @param list_of_lists List of lists to pad/stack.
    @param pad_value Value used to pad shorter rows.
    @return 2D NumPy array (dtype=int32).
    """
    if not list_of_lists:
        return np.array([], dtype=np.int32).reshape(0, 0)

    max_len = max(len(inner) for inner in list_of_lists)
    padded_lists = [list(inner) + [pad_value] * (max_len - len(inner)) for inner in list_of_lists]
    return np.array(padded_lists, dtype=np.int32)


def datetime_to_milliseconds_since_reference(arr, reference_str="2019-01-01T11:14:35.629"):
    """!Convert numpy datetime64 values to milliseconds since a reference timestamp.

    @param arr Array-like of ``np.datetime64`` values.
    @param reference_str Reference time ISO string (default "2019-01-01T11:14:35.629").
    @return Array of uint64 milliseconds since reference.
    """
    reference = np.datetime64(reference_str, 'ms')
    
    # Ensure array is datetime64[ms] for correct subtraction
    arr = np.asarray(arr).astype('datetime64[ms]')
    delta = arr - reference
    return delta.astype('timedelta64[ms]').astype('uint64')


def lonlat2xyz(lon, lat):
    """!Convert spherical lon/lat (radians) to unit-sphere Cartesian coordinates.

    @param lon Longitude(s) in radians.
    @param lat Latitude(s) in radians.
    @return Tuple ``(x, y, z)`` arrays/scalars.
    """
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

def find_stations_satellite_cif(lons, lats, timesteps, parameters, tree, decomp_domain, clon, number_of_cells, accepted_distance):
    """!Find satellite observation points (generic/cif) within the local PE domain.

    Uses nearest-cell search and inverse-distance weights; filters to owned cells and
    within an accepted great-circle distance.

    @param lons Longitudes of points to collocate.
    @param lats Latitudes of points to collocate.
    @param timesteps Observation datetimes per point.
    @param parameters Parameter names per point.
    @param tree KDTree on unit-sphere ICON cell centers.
    @param decomp_domain Ownership map per cell (0 = owned by this PE).
    @param clon Array whose shape defines (jc, jb) index layout.
    @param number_of_cells Number of cells to include in interpolation.
    @param accepted_distance Max accepted distance (km) to closest cell.
    @return Tuple:
        - jc_locs (int32)      : (N x number_of_cells)
        - jb_locs (int32)      : (N x number_of_cells)
        - lons_local (float64) : (N,)
        - lats_local (float64) : (N,)
        - timesteps_local      : (N,)
        - parameters_local (U10): (N,)
        - weights_all (float64): (N x number_of_cells)
    """
    # Define all lists as empty
    jc_locs = []
    jb_locs = []
    lons_local = []
    lats_local = []
    timesteps_local = []
    weights_all = []
    parameters_local = []

    # Loop thorugh every station
    for lon, lat, timestep, parameter in zip(lons, lats, timesteps, parameters):
        
        # Query the tree for the nearest cell
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = number_of_cells)
        closest_distance = dd[0][0] * 6371.0
        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0][0]] == 0 and closest_distance <= accepted_distance:
            jc_loc, jb_loc = np.unravel_index(ii[0], clon.shape) # Extract the indexes
            dd_local = dd[0]

            if number_of_cells == 1:
                jc_row = [jc_loc]
                jb_row = [jb_loc]
            else:
                jc_row = jc_loc.tolist()
                jb_row = jb_loc.tolist()
            weight_row = []

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
            while len(jc_row) < number_of_cells:
                jc_row.append(0)
                jb_row.append(0)
                weight_row.append(0.0)
            
            # Append all data, for the cells that were found in this PE's domain
            jc_locs.append(jc_row)
            jb_locs.append(jb_row)
            lons_local.append(lon)
            lats_local.append(lat)
            timesteps_local.append(timestep)
            weights_all.append(weight_row)
            parameters_local.append(parameter)

    # Return all data as numpy arrays
    return (np.array(jc_locs, dtype = np.int32),
            np.array(jb_locs, dtype = np.int32),
            np.array(lons_local, dtype = np.float64),
            np.array(lats_local, dtype = np.float64),
            np.array(timesteps_local), 
            np.array(parameters_local, dtype = 'U10'),
            np.array(weights_all, dtype = np.float64))


def find_stations_satellite_CH4(lons, lats, timesteps, tree, decomp_domain, clon, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree, accepted_distance, longitude_corners, latitude_corners, tree_corners, icon_polygons):
    """!Find CH4 satellite observation points and polygon overlaps in local PE domain.

    Combines nearest-cell search with polygon overlap (satellite footprint vs ICON cells)
    to compute horizontal weights. Also finds nearest CAMS index and time-fraction.

    @param lons, lats Positions of satellite ground pixels.
    @param timesteps Observation datetimes.
    @param tree KDTree on unit-sphere ICON cell centers.
    @param decomp_domain Ownership map per cell (0 = owned by this PE).
    @param clon Array whose shape defines (jc, jb) layout.
    @param pavg0_sat, pw_sat, ak_sat, qa0_sat Retrieval vectors/matrices per obs.
    @param cams_tree KDTree constructed on CAMS grid.
    @param accepted_distance Max accepted distance (km) to closest ICON cell.
    @param longitude_corners 2D array of pixel footprint longitudes (corners x obs).
    @param latitude_corners  2D array of pixel footprint latitudes (corners x obs).
    @param tree_corners STRtree of ICON polygons.
    @param icon_polygons List/array of ICON cell polygons.
    @return Tuple with jc/jb indices, positions/times, retrieval inputs, CAMS indices,
            CAMS fractions, weights, and covered area per obs.
    """
    # Define all lists as empty
    jc_locs = []
    jb_locs = []
    lons_local = []
    lats_local = []
    timesteps_local = []
    pavg0_local = []
    pw_local = []
    ak_local = []
    qa0_local = []
    cams_indices_local = []
    fracs_cams_local = []
    weights_all = []
    covered_areas = []

    # Loop through every station
    for i in range(len(lons)):
        lon = lons[i]
        lat = lats[i]
        timestep = timesteps[i]
        pavg0 = pavg0_sat[i]
        pw = pw_sat[i]
        ak = ak_sat[i]
        qa0 = qa0_sat[i]

        # Query the tree for the nearest cell
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = 1)
        closest_distance = dd[0] * 6371.0
        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0]] == 0 and closest_distance <= accepted_distance:
            jc_row = []
            jb_row = []
            weight_row = []
            jc_loc_reserve, jb_loc_reserve = np.unravel_index(ii[0], clon.shape) # Extract the indexes as reserve if the polygon matching didn't work
            tropomi_polygon = Polygon(zip(np.deg2rad(longitude_corners[:, i]), np.deg2rad(latitude_corners[:, i])))

            possible_matches = tree_corners.query(tropomi_polygon)
            covered_area = 0.0
            for match in possible_matches:
                if tropomi_polygon.intersects(icon_polygons[match]):
                    overlap_area = tropomi_polygon.intersection(icon_polygons[match]).area
                    # if a ICON cell is at least covered by some percentage with the obs, then count this cell
                    if icon_polygons[match].area > 0.0 and overlap_area > 0.0:
                        jc_r, jb_r = np.unravel_index(match, clon.shape)
                        covered_area += overlap_area / tropomi_polygon.area
                        jc_row.append(jc_r)
                        jb_row.append(jb_r)
                        weight_row.append(overlap_area / tropomi_polygon.area)

            # If the polygon matching didn't work, just take the nearest cell
            if jc_row == []:
                jc_row = [jc_loc_reserve]
                jb_row = [jb_loc_reserve]
                weight_row = [1.0]


            # Get sorted indices based on weights (descending)
            sorted_indices = sorted(range(len(weight_row)), key=lambda i: weight_row[i], reverse=True)

            # Apply the sorting to each list
            jc_row = [jc_row[i] for i in sorted_indices]
            jb_row = [jb_row[i] for i in sorted_indices]
            weight_row = [weight_row[i] for i in sorted_indices]


            if weight_row != [] and np.sum(weight_row) != 0:
                weight_row = weight_row / np.sum(weight_row)


            # Compute frac_cams and cams_index for later appending cams data to the model data
            cams_distances, cams_index = cams_tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = 1)
            cams_prev = timestep.replace(hour=(timestep.hour // 6) * 6, minute=0, second=0, microsecond=0)
            cams_next = cams_prev + datetimelib.timedelta(hours=6)
            frac_cams = (timestep - cams_prev).total_seconds() / (cams_next - cams_prev).total_seconds()            

            # Append all data, for the cells that were found in this PE's domain
            jc_locs.append(jc_row)
            jb_locs.append(jb_row)
            lons_local.append(lon)
            lats_local.append(lat)
            timesteps_local.append(timestep)
            pavg0_local.append(pavg0)
            pw_local.append(pw)
            ak_local.append(ak)
            qa0_local.append(qa0)
            cams_indices_local.append(cams_index)
            fracs_cams_local.append(frac_cams)
            weights_all.append(weight_row)
            covered_areas.append(covered_area)

    # Pad all of the potentially different lengths of lists
    jc_locs = pad_list_of_lists_int(jc_locs, pad_value=0)
    jb_locs = pad_list_of_lists_int(jb_locs, pad_value=0)
    weights_all = pad_list_of_lists_float(weights_all, pad_value=0.0)

    # Return all data as numpy arrays
    return (jc_locs,
            jb_locs,
            np.array(lons_local, dtype = np.float64),
            np.array(lats_local, dtype = np.float64),
            np.array(timesteps_local), 
            np.array(pavg0_local, dtype = np.float64),
            np.array(pw_local, dtype = np.float64), 
            np.array(ak_local, dtype = np.float64), 
            np.array(qa0_local, dtype = np.float64), 
            np.array(cams_indices_local, dtype = np.int32), 
            np.array(fracs_cams_local, dtype = np.float64), 
            weights_all, 
            np.array(covered_areas))

def write_header_sat(comm, file_name):
    """!Create NetCDF header for CH4 satellite output.

    Defines unlimited dimension 'index' and variables: date, lon, lat, CH4, covered_area.

    @param comm MPI communicator (root-only writes).
    @param file_name Output file path to create.
    @return None.
    """
    if comm.Get_rank() == 0:
        ncfile_sat = Dataset(file_name, 'w', format='NETCDF4')
        index = ncfile_sat.createDimension('index', None)
        date = ncfile_sat.createVariable('date', 'u8',('index',) )
        date.units = "milliseconds since 2019-01-01 11:14:35.629000" 
        date.calendar = "proleptic_gregorian"
        lon = ncfile_sat.createVariable('lon', 'f8',('index',) )
        lat = ncfile_sat.createVariable('lat', 'f8',('index',) )
        ch4 = ncfile_sat.createVariable('CH4', 'f8',('index',) )
        covered_area = ncfile_sat.createVariable('covered_area', 'f8', ('index',) )
        ncfile_sat.close()
    
def write_header_sat_cif(comm, file_name, num_levels):
    """!Create NetCDF header for satellite full-column (cif) output.

    Defines dims 'index' (unlimited) and 'level', and variables:
    date, lon, lat, parameter, measurement(index, level).

    @param comm MPI communicator (root-only writes).
    @param file_name Output file path to create.
    @param num_levels Number of vertical levels.
    @return None.
    """
    if comm.Get_rank() == 0:
        ncfile_sat = Dataset(file_name, 'w', format='NETCDF4')
        index = ncfile_sat.createDimension('index', None)
        level = ncfile_sat.createDimension('level', num_levels)
        date = ncfile_sat.createVariable('date', 'u8',('index',) )
        date.units = "milliseconds since 2019-01-01 00:00:00" 
        date.calendar = "proleptic_gregorian"
        lon = ncfile_sat.createVariable('lon', 'f8',('index',) )
        lat = ncfile_sat.createVariable('lat', 'f8',('index',) )
        parameter = ncfile_sat.createVariable('parameter', 'S10',('index',) )
        measurement = ncfile_sat.createVariable('measurement', 'f8',('index','level') )
        ncfile_sat.close()

def write_satellite_cif(comm, done_data: SatelliteDataDoneGeneral, file_name_output):
    """!Append completed satellite column data (generic/cif) to NetCDF.

    Gathers from ranks, sorts by date, converts times to milliseconds since
    2019-01-01 00:00:00, and appends.

    @param comm MPI communicator.
    @param done_data Completed data container (SatelliteDataDoneGeneral).
    @param file_name_output Existing NetCDF file path (with header).
    @return None.
    """
    done_data_local = None
    done_counter = done_data.counter

    # Collect the local point data, that we want to write out
    if done_counter > 0:
        done_data_local = {
            "lon": done_data.lon[:done_counter],
            "lat": done_data.lat[:done_counter],
            "date": done_data.timestep[:done_counter],
            "measurement": done_data.measurement[:done_counter],
            "parameter": done_data.parameter[:done_counter],
        }
    # Gather the local data to root 0, such that one process has all data that needs to be written out
    gathered_done_data = comm.gather(done_data_local, root=0)

    # The rank that has gathered the data will now write it out
    if comm.Get_rank() == 0:
        final_data = {
            "lon": [],
            "lat": [],
            "date": [],
            "measurement": [],
            "parameter": [],
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
            sort_indices = np.argsort(final_data["date"])
            for key in final_data:
                final_data[key] = final_data[key][sort_indices]
            ncfile = Dataset(file_name_output, 'a')
            obs_index = ncfile.variables['lon'].shape[0]
            new_points = final_data['lon'].shape[0]

            final_data['date'] = datetime_to_milliseconds_since_reference(final_data['date'], reference_str="2019-01-01T00:00:00")

            for var_name, data in final_data.items():
                    if var_name != 'measurement':
                        ncfile.variables[var_name][obs_index: obs_index + new_points] = data
                    else:
                        ncfile.variables[var_name][obs_index: obs_index + new_points, :] = data
            
            # Write out the data
            ncfile.close()
            
    done_data.counter = 0 # Reset the done counter

def write_satellite_CH4(comm, done_data: SatelliteDataDoneCH4, file_name_output):
    """!Append completed CH4 satellite comparisons to NetCDF.

    Gathers from ranks, sorts by date, converts times to milliseconds since
    default reference, and appends lon/lat/date/CH4/covered_area.

    @param comm MPI communicator.
    @param done_data Completed data container (SatelliteDataDoneCH4).
    @param file_name_output Existing NetCDF file path (with header).
    @return None.
    """
    done_data_local = None
    done_counter = done_data.counter
    # Collect the local point data, that we want to write out
    if done_counter > 0:
        done_data_local = {
            "lon": done_data.lon[:done_counter],
            "lat": done_data.lat[:done_counter],
            "date": done_data.timestep[:done_counter],
            "CH4": done_data.CH4[:done_counter],
            "covered_area": done_data.covered_areas[:done_counter],
        }

    # Gather the local data to root 0, such that one process has all data that needs to be written out
    gathered_done_data = comm.gather(done_data_local, root=0)

    # The rank that has gathered the data will now write it out
    if comm.Get_rank() == 0:
        final_data = {
            "lon": [],
            "lat": [],
            "date": [],
            "CH4": [],
            "covered_area": [],
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
            sort_indices = np.argsort(final_data["date"])
            for key in final_data:
                final_data[key] = final_data[key][sort_indices]
            ncfile = Dataset(file_name_output, 'a')
            obs_index = ncfile.variables['lon'].shape[0]
            new_points = final_data['lon'].shape[0]

            final_data['date'] = datetime_to_milliseconds_since_reference(final_data['date'])

            for var_name, data in final_data.items():
                    ncfile.variables[var_name][obs_index: obs_index + new_points] = data
            
            # Write out the data
            ncfile.close()
            
    done_data.counter = 0 # Reset the done counter

def read_in_satellite_data_CH4(comm, tree, decomp_domain, clon, start_model, end_model, tropomi_filename, cams_base_path, cams_params_file, accepted_distance, icon_polygons):
    """!Read CH4 satellite inputs, CAMS metadata, and build to-do/done containers.

    Filters TROPOMI observations to the model window, prepares CAMS files and
    hybrid coefficients, builds KD trees and STRtree, finds local stations, and
    allocates ``SatelliteDataToDoCH4`` / ``SatelliteDataDoneCH4``.

    @param comm MPI communicator.
    @param tree KDTree on ICON grid (unit-sphere).
    @param decomp_domain Ownership map per cell (0 = owned).
    @param clon Array whose shape defines (jc, jb) layout.
    @param start_model Experiment start datetime.
    @param end_model Experiment end datetime.
    @param tropomi_filename Path to TROPOMI NetCDF.
    @param cams_base_path Base directory of CAMS files.
    @param cams_params_file Path to CAMS parameter NetCDF (hybrid coeffs).
    @param accepted_distance Max accepted distance (km) to closest ICON cell.
    @param icon_polygons List/array of ICON polygons.
    @return Tuple ``(local_data_to_do, local_data_done, cams_files_dict)``.
    """
    cams_files_dict = {} # Dictionary for the cams file names

    # Create the dictionary for the cams file names
    dt = start_model.replace(hour=(start_model.hour // 6))
    while dt <= end_model:
        dt_str = dt.strftime("%Y%m%d%H")
        if dt.month == 1:
            fname = f"cams73_v22r2_ch4_conc_surface_inst_{dt_str}.nc"
        else:
            fname = f"cams73_v22r2_ch4_conc_surface_inst_{dt_str}_lbc.nc"
        fname = f"cams73_v22r2_ch4_conc_surface_inst_{dt_str}.nc"
        full_path = os.path.join(cams_base_path, fname)
        cams_files_dict[dt] = full_path
        dt += datetimelib.timedelta(hours=6)


    if comm.Get_rank() == 0:
        tropomi_ds = xr.open_dataset(tropomi_filename)
        # Extract all the needed values
        times_sat = tropomi_ds['date'].values
        satellite_lons = tropomi_ds['lon'].values
        satellite_lats = tropomi_ds['lat'].values
        pavg0_sat = tropomi_ds['pavg0'].values
        qa0_sat = tropomi_ds['qa0'].values
        ak_sat = tropomi_ds['ak'].values
        pw_sat = tropomi_ds['pw'].values
        latitude_corners = tropomi_ds['latitude_bounds'].values
        longitude_corners = tropomi_ds['longitude_bounds'].values
        obs_time_dts = pd.to_datetime(times_sat)
        
        # Only keep the points, where the time is between the starting and ending time
        valid_mask = (obs_time_dts >= start_model) & (obs_time_dts <= end_model)
        satellite_lons = satellite_lons[valid_mask]
        satellite_lats = satellite_lats[valid_mask]
        pavg0_sat = pavg0_sat[valid_mask]
        pw_sat = pw_sat[valid_mask]
        ak_sat = ak_sat[valid_mask]
        qa0_sat = qa0_sat[valid_mask]
        obs_time_dts = obs_time_dts[valid_mask]
        latitude_corners = latitude_corners[: , valid_mask]
        longitude_corners = longitude_corners[:, valid_mask]


        # Load one example CAMS file, for computing the correct CAMS indices. This supposes that the CAMS data is ordered the same way always
        cams_example_time = start_model.replace(hour=(start_model.hour // 6) * 6, minute=0, second=0, microsecond=0)
        example_file = cams_files_dict[cams_example_time]
        cams_ds = xr.open_dataset(example_file)
        cams_clon = np.asarray(cams_ds["clon"])
        cams_clat = np.asarray(cams_ds["clat"])

        # Load a CAMS parameter file with parameters needed later
        cams_param_ds = xr.open_dataset(cams_params_file)
        hyam = cams_param_ds["hyam"].values
        hybm = cams_param_ds["hybm"].values
        hyai = cams_param_ds["hyai"].values
        hybi = cams_param_ds["hybi"].values

        # Reverse CAMS hybrid coefficients so that they go from TOA to surface
        hyam = hyam[::-1]
        hybm = hybm[::-1]
        hyai = hyai[::-1]
        hybi = hybi[::-1]
    else:
        satellite_lons = None
        satellite_lats = None
        obs_time_dts = None
        pavg0_sat = None
        pw_sat = None
        ak_sat = None
        qa0_sat = None
        cams_clon = None
        cams_clat = None
        hyam = None
        hybm = None
        hyai = None
        hybi = None
        longitude_corners = None
        latitude_corners = None

    # Broadcast the data to all processes, from root 0
    satellite_lons = comm.bcast(satellite_lons, root = 0)
    satellite_lats = comm.bcast(satellite_lats, root = 0)
    obs_time_dts = comm.bcast(obs_time_dts, root = 0)
    pavg0_sat = comm.bcast(pavg0_sat, root = 0)
    pw_sat = comm.bcast(pw_sat, root = 0)
    ak_sat = comm.bcast(ak_sat, root = 0)
    qa0_sat = comm.bcast(qa0_sat, root = 0)
    cams_clat = comm.bcast(cams_clat, root = 0)
    cams_clon = comm.bcast(cams_clon, root = 0)
    hyam = comm.bcast(hyam, root = 0)
    hybm = comm.bcast(hybm, root = 0)
    hyai = comm.bcast(hyai, root = 0)
    hybi = comm.bcast(hybi, root = 0)
    longitude_corners = comm.bcast(longitude_corners, root = 0)
    latitude_corners = comm.bcast(latitude_corners, root = 0)

    # Create the cams tree for searching later
    cams_xyz = np.c_[lonlat2xyz(cams_clon, cams_clat)]
    cams_tree = KDTree(cams_xyz)

    # Create the icon polygon tree
    tree_corners = STRtree(icon_polygons)

    (jc_loc_satellite, jb_loc_satellite, satellite_lons, satellite_lats, satellite_timestep, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_indices_sat, fracs_cams, weights_sat, covered_areas) = find_stations_satellite_CH4(satellite_lons, satellite_lats, obs_time_dts, tree, decomp_domain, clon, pavg0_sat, pw_sat, ak_sat, qa0_sat, cams_tree, accepted_distance, longitude_corners, latitude_corners, tree_corners, icon_polygons)
   
    N_satellite_points = satellite_lons.shape[0] # Amount of satellite points in the local PE

    # Initialize all needed arrays as empty arrays of correct size
    done_lons_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_lats_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_times_sat = np.empty(N_satellite_points, dtype='datetime64[ns]')
    done_CH4_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_covered_areas = np.empty(N_satellite_points, dtype=np.float64)
    done_counter_sat = 0

    local_data_to_do = SatelliteDataToDoCH4(satellite_lons, satellite_lats, satellite_timestep, jc_loc_satellite, jb_loc_satellite, weights_sat, qa0_sat, ak_sat, pw_sat, pavg0_sat, cams_indices_sat, fracs_cams, hyam, hybm, hyai, hybi, covered_areas)
    local_data_done = SatelliteDataDoneCH4(done_lons_sat, done_lats_sat, done_times_sat, done_CH4_sat, done_counter_sat, done_covered_areas)
    
    return local_data_to_do, local_data_done, cams_files_dict # Return all of the Data

def read_in_satellite_data_cif(comm, tree, decomp_domain, clon, start_model, end_model, path_to_csv, number_of_cells, accepted_distance, N_levels):
    """!Read generic satellite inputs (from CSV) and build to-do/done containers.

    De-duplicates, filters to model window, finds local stations via
    ``find_stations_satellite_cif``, allocates
    ``SatelliteDataToDoGeneral`` / ``SatelliteDataDoneGeneral``.

    @param comm MPI communicator.
    @param tree KDTree on ICON grid (unit-sphere).
    @param decomp_domain Ownership map per cell (0 = owned).
    @param clon Array whose shape defines (jc, jb) layout.
    @param start_model Experiment start datetime.
    @param end_model Experiment end datetime.
    @param path_to_csv Input CSV path.
    @param number_of_cells Number of cells to include in interpolation.
    @param accepted_distance Max accepted distance (km) to closest ICON cell.
    @param N_levels Number of vertical levels (for output allocation).
    @return Tuple ``(local_data_to_do, local_data_done)``.
    """
    if comm.Get_rank() == 0:
        df = pd.read_csv(path_to_csv, sep=',')
        df = df.dropna(subset=['lon', 'lat', 'tstep', 'parameter'])
        df = df.drop_duplicates(subset=['tstep', 'lon', 'lat', 'parameter']).reset_index(drop=True)
        
        df['datetime'] = pd.to_datetime(df['tstep'], unit='h', origin = start_model)

        # convert the needed data to numpy arrays
        lons = df['lon'].to_numpy()
        lats = df['lat'].to_numpy()
        timesteps = df['datetime'].to_numpy()
        parameters = df['parameter'].to_numpy()
        
        # Only keep the points, where the time is between the starting and ending time
        valid_mask = (timesteps >= start_model) & (timesteps <= end_model)
        lons = lons[valid_mask]
        lats = lats[valid_mask]
        timesteps = timesteps[valid_mask]

    else:
        lons = None
        lats = None
        timesteps = None
        parameters = None

    # Broadcast the data to all processes, from root 0
    lons = comm.bcast(lons, root = 0)
    lats = comm.bcast(lats, root = 0)
    timesteps = comm.bcast(timesteps, root = 0)
    parameters = comm.bcast(parameters, root = 0)

    (jc_loc_satellite, jb_loc_satellite, satellite_lons, satellite_lats, satellite_timestep, satellite_parameters, weights_all) = find_stations_satellite_cif(lons, lats, timesteps, parameters, tree, decomp_domain, clon, number_of_cells, accepted_distance)
   
    N_satellite_points = satellite_lons.shape[0] # Amount of satellite points in the local PE
    # Initialize all needed arrays as empty arrays of correct size
    done_lons_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_lats_sat = np.empty(N_satellite_points, dtype=np.float64)
    done_times_sat = np.empty(N_satellite_points, dtype='datetime64[ns]')
    done_measurements_sat = np.empty((N_satellite_points, N_levels), dtype=np.float64)
    done_parameters_sat = np.empty(N_satellite_points, dtype='U10')
    done_counter_sat = 0

    local_data_to_do = SatelliteDataToDoGeneral(satellite_lons, satellite_lats, satellite_timestep, jc_loc_satellite, jb_loc_satellite, satellite_parameters, weights_all)
    local_data_done = SatelliteDataDoneGeneral(done_lons_sat, done_lats_sat, done_times_sat, done_parameters_sat, done_measurements_sat, done_counter_sat)
    
    return local_data_to_do, local_data_done # Return all of the Data

def update_cams(datetime, cams_files_dict, cams_prev_data=None, cams_next_data=None):
    """!Update/open CAMS datasets for the current time window (6-hour cadence).

    Closes previously opened datasets (if any) and opens the pair bracketing
    ``datetime`` (prev and next).

    @param datetime Current datetime.
    @param cams_files_dict Mapping from datetime -> CAMS filename.
    @param cams_prev_data Previously opened xarray Dataset (or None).
    @param cams_next_data Previously opened xarray Dataset (or None).
    @return Tuple ``(cams_prev_data, cams_next_data)`` as xarray Datasets.
    """
    cams_prev_time = pd.to_datetime(datetime)
    cams_next_time = cams_prev_time + datetimelib.timedelta(hours=6)
    if cams_prev_data is not None:
        cams_prev_data.close()
    if cams_next_data is not None:
        cams_next_data.close()
    cams_prev_data = xr.open_dataset(cams_files_dict[cams_prev_time])
    cams_next_data = xr.open_dataset(cams_files_dict[cams_next_time])
    return cams_prev_data, cams_next_data

def tracking_satellite_cif_same_index(datetime, data_to_do: SatelliteDataToDoGeneral, data_done: SatelliteDataDoneGeneral, data_np, dict_vars, operations_dict, num_levels):
    """!Compute satellite columns (cif) using same-index vertical sampling.

    Horizontally interpolates via inverse-distance weights; vertically samples
    each neighbor at the same index as the closest column, then averages.

    @param datetime Current datetime.
    @param data_to_do Pending container (SatelliteDataToDoGeneral).
    @param data_done Completed container (SatelliteDataDoneGeneral).
    @param data_np Mapping ``{var: list[np.ndarray]}`` of model fields.
    @param dict_vars Variable recipe dict with ``factor`` and ``signs``.
    @param operations_dict Mapping sign string -> binary operator.
    @param num_levels Number of vertical levels to output.
    @return None.
    """
    if data_to_do.timestep.size > 0: # Checks if there is still work to do

        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is greater or equal to the moment we want to measure. They are ready for measurement
        ready_mask = data_to_do.timestep <= model_time_np

        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready_sat = data_to_do.jc_loc[ready_mask]
            jb_ready_sat = data_to_do.jb_loc[ready_mask]

            num_ready = np.sum(ready_mask)

            variables = data_to_do.parameter[ready_mask]
            for i, variable in enumerate(variables):
                jc = jc_ready_sat[i]
                jb = jb_ready_sat[i]
                done_counter = data_done.counter
                list = data_np[variable]
                ICON_profile = list[0][jc, :, jb, 0, 0] * dict_vars[variable]['factor'][0]
                result = np.empty(num_levels, dtype = np.float64)
                for sign, j in zip(dict_vars[variable]['signs'], range(1, len(list))):
                    ICON_profile = operations_dict[sign](ICON_profile, list[j][jc, :, jb, 0, 0] * dict_vars[variable]['factor'][i])
                for j in range(num_levels):
                    result[j] = data_to_do.weights[ready_mask][i][0] * ICON_profile[0, j]

                    for k in range(1, len(jb)):
                        best_index = j
                        result[j] += data_to_do.weights[ready_mask][i][j] * ICON_profile[k, best_index]

                # Repeat spatial info for each level
                data_done.lon[done_counter:done_counter + 1] = data_to_do.lon[ready_mask][i]
                data_done.lat[done_counter:done_counter + 1] = data_to_do.lat[ready_mask][i]
                data_done.timestep[done_counter:done_counter + 1] = data_to_do.timestep[ready_mask][i]
                data_done.measurement[done_counter:done_counter + 1 , :] = result
                data_done.parameter[done_counter:done_counter + 1] = variable

            data_done.counter += 1

            

            # Only keep the satellite points that aren't done yet
            keep_mask = ~ready_mask

            data_to_do.apply_mask(keep_mask)
        

def tracking_satellite_cif_pressures(datetime, data_to_do: SatelliteDataToDoGeneral, data_done: SatelliteDataDoneGeneral, data_np, dict_vars, operations_dict, pres_np, num_levels):
    """!Compute satellite columns (cif) using pressure-based vertical interpolation.

    For each neighbor column, find closest pressure to the target profile,
    perform linear interpolation in pressure space, then horizontally average.

    @param datetime Current datetime.
    @param data_to_do Pending container (SatelliteDataToDoGeneral).
    @param data_done Completed container (SatelliteDataDoneGeneral).
    @param data_np Mapping ``{var: list[np.ndarray]}`` of model fields.
    @param dict_vars Variable recipe dict with ``factor`` and ``signs``.
    @param operations_dict Mapping sign string -> binary operator.
    @param pres_np 3D array of model pressures (Pa) at mid-levels.
    @param num_levels Number of vertical levels to output.
    @return None.
    """
    if data_to_do.timestep.size > 0: # Checks if there is still work to do

        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is greater or equal to the moment we want to measure. They are ready for measurement
        ready_mask = data_to_do.timestep <= model_time_np

        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready_sat = data_to_do.jc_loc[ready_mask]
            jb_ready_sat = data_to_do.jb_loc[ready_mask]

            num_ready = np.sum(ready_mask)

            variables = data_to_do.parameter[ready_mask]
            for i in range(num_ready):
                for variable in variables:
                    jc = jc_ready_sat[i]
                    jb = jb_ready_sat[i]
                    done_counter = data_done.counter
                    list = data_np[variable]
                    ICON_profile = list[0][jc, :, jb, 0, 0] * dict_vars[variable]['factor'][0]
                    result = np.empty(num_levels, dtype = np.float64)
                    for sign, j in zip(dict_vars[variable]['signs'], range(1, len(list))):
                        ICON_profile = operations_dict[sign](ICON_profile, list[j][jc, :, jb, 0, 0] * dict_vars[variable]['factor'][j])
                    pres = (pres_np[jc, :, jb].squeeze()) / 1.e2
                    target_pres = pres_np[jc[0], :, jb[0]].squeeze()
                    for j in range(num_levels):
                        this_target_pressure = target_pres[j]
                        result[j] = data_to_do.weights[ready_mask][i][0] * ICON_profile[0, j]

                        for k in range(1, len(jb)):
                            best_index = int(np.argmin(np.abs(pres_np[jc[k], :, jb[k]] - this_target_pressure)))

                            actual_pressure_closest = pres_np[jc[k], best_index, jb[k]]
                            second_index = best_index
                            # Second index is for height interpolation. depending on where the closest height is, compute the second index, also taking into consideration boundaries
                            if actual_pressure_closest >= this_target_pressure and actual_pressure_closest != pres_np[jc[k], -1, jb[k]]:
                                second_index += 1
                            elif actual_pressure_closest < this_target_pressure and actual_pressure_closest != pres_np[jc[k], 0, jb[k]]:
                                second_index -= 1

                            second_pres = pres_np[jc[k], second_index, jb[k]]
                            vertical_weight = 0
                            if second_pres - actual_pressure_closest != 0:
                                vertical_weight = (this_target_pressure - actual_pressure_closest) / (second_pres - actual_pressure_closest)

                            this_cell_interpolation_vertical = ICON_profile[k , best_index] + vertical_weight * (ICON_profile[k, second_index] - ICON_profile[k, best_index])
                            result[j] += data_to_do.weights[ready_mask][i][k] * this_cell_interpolation_vertical

                    # Repeat spatial info for each level
                    data_done.lon[done_counter:done_counter + 1] = data_to_do.lon[ready_mask][i]
                    data_done.lat[done_counter:done_counter + 1] = data_to_do.lat[ready_mask][i]
                    data_done.timestep[done_counter:done_counter + 1] = data_to_do.timestep[ready_mask][i]
                    data_done.measurement[done_counter:done_counter + 1 , :] = result
                    data_done.parameter[done_counter:done_counter + 1] = variable
                    data_done.counter += 1

            

            # Only keep the satellite points that aren't done yet
            keep_mask = ~ready_mask

            data_to_do.apply_mask(keep_mask)
        


def tracking_CH4_satellite(datetime, CH4_EMIS_np, CH4_BG_np, pres_ifc_np, pres_np, data_to_do: SatelliteDataToDoCH4, data_done: SatelliteDataDoneCH4, cams_prev_data, cams_next_data):
    """!Compute CH4 total column for satellite comparison.

    Steps:
      1) Build ICON CH4 profile = BG + scaled EMIS contribution (per level).
      2) Pressure-based vertical interpolation to target column profile.
      3) Horizontal averaging via satellite footprint coverage weights.
      4) Interpolate CAMS in time and extend low-pressure part of the profile.
      5) Apply retrieval operators (pw, ak, qa0, pavg0) to get total column.

    @param datetime Current datetime.
    @param CH4_EMIS_np ICON CH4 emission field (per level).
    @param CH4_BG_np ICON CH4 background field (per level).
    @param pres_ifc_np Pressures at interfaces (Pa).
    @param pres_np Pressures at mid-levels (Pa).
    @param data_to_do Pending container (SatelliteDataToDoCH4).
    @param data_done Completed container (SatelliteDataDoneCH4).
    @param cams_prev_data xarray Dataset at previous CAMS step.
    @param cams_next_data xarray Dataset at next CAMS step.
    @return None.
    """
    if data_to_do.timestep.size > 0: # Checks if there is still work to do

        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is greater or equal to the moment we want to measure. They are ready for measurement
        ready_mask = data_to_do.timestep <= model_time_np

        if np.any(ready_mask):
            # Filter arrays for ready stations
            jc_ready_sat = data_to_do.jc_loc[ready_mask, :]
            jb_ready_sat = data_to_do.jb_loc[ready_mask, :]
            frac_cams_ready = data_to_do.frac_cams[ready_mask]
            frac_cams_ready = frac_cams_ready[:, np.newaxis]
            cams_indices_ready = data_to_do.cams_index[ready_mask]

            ## In general the following applies. All data is from TOA to surface. Up until the get coef function, which expects the data the other way around
            ## So there we turn everything around
            
            # Extract the pressure and CAMS data. For the CAMS data interpolate linearly between the 6 hour intervals

            pb_mod = (pres_ifc_np[jc_ready_sat[:, 0], :, jb_ready_sat[:, 0]].squeeze()) / 1.e2
            pb_mod_mc =(pres_np[jc_ready_sat[:, 0], :, jb_ready_sat[:, 0]].squeeze()) / 1.e2

            cams_indices_ready = cams_indices_ready.flatten()
            CAMS_obs_prev = cams_prev_data["CH4"].isel(time=0, cell = cams_indices_ready).values[::-1].T
            CAMS_obs_next = cams_next_data["CH4"].isel(time=0, cell = cams_indices_ready).values[::-1].T

            CAMS_aps_prev = cams_prev_data["ps"].isel(time=0, cell = cams_indices_ready).values
            CAMS_aps_next = cams_next_data["ps"].isel(time=0, cell = cams_indices_ready).values

            CAMS_aps_prev_reshaped = CAMS_aps_prev[:, np.newaxis]
            CAMS_aps_next_reshaped = CAMS_aps_next[:, np.newaxis]
            N_ready = CAMS_aps_prev.shape[0]  # number of ready satellite points

            hyam_new_axis = np.tile(data_to_do.hyam, (N_ready, 1))
            hybm_new_axis = np.tile(data_to_do.hybm, (N_ready, 1))
            hyai_new_axis = np.tile(data_to_do.hyai, (N_ready, 1))
            hybi_new_axis = np.tile(data_to_do.hybi, (N_ready, 1))
            # This is a formula I got from the CAMS data
            CAMS_p_prev =  (hyam_new_axis + hybm_new_axis * CAMS_aps_prev_reshaped) / 1.e2
            CAMS_p_next = (hyam_new_axis + hybm_new_axis * CAMS_aps_next_reshaped) / 1.e2
            CAMS_pressures = (1. - frac_cams_ready) * CAMS_p_prev + frac_cams_ready * CAMS_p_next

            CAMS_i_prev = (hyai_new_axis + hybi_new_axis * CAMS_aps_prev_reshaped) / 1.e2
            CAMS_i_next =  (hyai_new_axis + hybi_new_axis * CAMS_aps_next_reshaped) / 1.e2
            CAMS_interfaces = (1. - frac_cams_ready) * CAMS_i_prev + frac_cams_ready * CAMS_i_next
        
            num_ready = np.sum(ready_mask)
            num_levels = len(CH4_BG_np[0, :, 0].squeeze())
            ICON_profile = np.empty((num_ready, num_levels), dtype = np.float64)
            for i in range(num_ready):
                    jc = jc_ready_sat[i, :]
                    jb = jb_ready_sat[i, :]
                    profiles = (
                        (Mda / MCH4) * CH4_BG_np[jc, :, jb].squeeze() \
                        + 1.e9 * (Mda / MCH4) * CH4_EMIS_np[jc, :, jb].squeeze()
                    )
                
                    result = np.empty(num_levels, dtype = np.float64)
                    pres = (pres_np[jc, :, jb].squeeze()) / 1.e2
                    target_pres = pres_np[jc[0], :, jb[0]].squeeze()
                    for j in range(num_levels):
                        this_target_pressure = target_pres[j]
                        result[j] = data_to_do.weights[ready_mask][i, 0] * profiles[0, j]

                        for k in range(1, len(jb)):
                            best_index = int(np.argmin(np.abs(pres_np[jc[k], :, jb[k]] - this_target_pressure)))

                            actual_pressure_closest = pres_np[jc[k], best_index, jb[k]]
                            second_index = best_index
                            # Second index is for vertical interpolation. depending on where the closest pressure is, compute the second index, also taking into consideration boundaries
                            if actual_pressure_closest >= this_target_pressure and actual_pressure_closest != pres_np[jc[k], -1, jb[k]]:
                                second_index += 1
                            elif actual_pressure_closest < this_target_pressure and actual_pressure_closest != pres_np[jc[k], 0, jb[k]]:
                                second_index -= 1

                            second_pres = pres_np[jc[k], second_index, jb[k]]
                            vertical_weight = 0
                            if second_pres - actual_pressure_closest != 0:
                                vertical_weight = (this_target_pressure - actual_pressure_closest) / (second_pres - actual_pressure_closest)

                            this_cell_interpolation_vertical = profiles[k , best_index] + vertical_weight * (profiles[k, second_index] - profiles[k, best_index])
                            result[j] += data_to_do.weights[ready_mask][i][k] * this_cell_interpolation_vertical
                    ICON_profile[i, :] = result
            
            # Linearly interpolate the CAMS data to the current timestep, as CAMS data is only every 6 hours
            CAMS_obs = (1. - frac_cams_ready) * CAMS_obs_prev + frac_cams_ready * CAMS_obs_next

            ## The following code takes the ICON data and extends it by the CAMS data, as the ICON data does not go to very low pressures
            CAMS_profile = np.array([
                CAMS_obs[i, CAMS_pressures[i] < np.min(pb_mod_mc[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            pb_cams_mc = np.array([
                CAMS_pressures[i, CAMS_pressures[i] < np.min(pb_mod_mc[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            pb_cams = np.array([
                CAMS_interfaces[i, CAMS_interfaces[i] < np.min(pb_mod[i])]
                for i in range(CAMS_obs.shape[0])
            ], dtype=object)
            # Was sometimes buggy, added this for safety
            ICON_profile = np.atleast_2d(ICON_profile)
            CAMS_profile = np.atleast_2d(CAMS_profile)
            pb_mod = np.atleast_2d(pb_mod)
            pb_cams = np.atleast_2d(pb_cams)
            pb_mod_mc = np.atleast_2d(pb_mod_mc)
            pb_cams_mc = np.atleast_2d(pb_cams_mc)
            
            tracer_profile = np.array([
                np.concatenate([CAMS_profile[i], ICON_profile[i]])
                for i in range(len(CAMS_profile))
            ], dtype=object)

            pb_profile = np.array([
                np.concatenate([pb_cams[i], pb_mod[i]])
                for i in range(len(pb_cams))
            ], dtype=object)

            pb_mc_profile = np.array([
                np.concatenate([pb_cams_mc[i], pb_mod_mc[i]])
                for i in range(len(pb_cams_mc))
            ], dtype=object)

            # Now as we get to the get coef funciton we need to turn around all of the data to: from surface to TOA
            # Also the computation afterwards is based on Michael Steiners computation
            pb_ret = data_to_do.pavg0[ready_mask]
            coef_matrix = []
            # pb_ret = pb_ret[:, ::-1]
            pb_profile = pb_profile[:, ::-1]
            tracer_profile = tracer_profile[:, ::-1]
            for i in range(pb_ret.shape[0]):
                # print(f"pb_ret.shape: {pb_ret[i].shape}, pb_profile.shape: {pb_profile[i].shape}, pb_ret: {pb_ret[i]}, pb_profile: {pb_profile[i]}", file=sys.stderr)
                coefs = get_int_coefs(pb_ret[i], pb_profile[i])

                coef_matrix.append(coefs)

            coef_matrix = np.array(coef_matrix)

            pwf = data_to_do.pw[ready_mask]
            # pwf = pwf[:, ::-1]
            averaging_kernel = data_to_do.ak[ready_mask]
            # averaging_kernel = averaging_kernel[:, ::-1]
            important_stuff = data_to_do.qa0[ready_mask]
            # important_stuff = important_stuff[:, ::-1]

            if pb_ret[0, 0] < pb_ret[0, -1]:
                pb_ret = pb_ret[:, ::-1]
                pwf = pwf[:, ::-1]
                averaging_kernel = averaging_kernel[:, ::-1]
                important_stuff = important_stuff[:, ::-1]

            avpw = pwf * averaging_kernel
            prior_col = np.sum(pwf * important_stuff, axis=1)

            profile_intrp = np.matmul(coef_matrix, tracer_profile[..., np.newaxis])[..., 0]
            tc = prior_col + np.sum(avpw * (profile_intrp - important_stuff), axis=1)


            num_ready = np.sum(ready_mask) # Count how many points are ready
            done_counter = data_done.counter

            # Add all of the done points to the done arrays
            data_done.lon[done_counter:done_counter + num_ready] = data_to_do.lon[ready_mask]
            data_done.lat[done_counter:done_counter + num_ready] = data_to_do.lat[ready_mask]
            data_done.timestep[done_counter:done_counter + num_ready] = data_to_do.timestep[ready_mask]
            data_done.CH4[done_counter:done_counter + num_ready] = tc
            data_done.covered_areas[done_counter:done_counter + num_ready] = data_to_do.covered_areas[ready_mask]

            # Keep count of how many satellite points are done
            data_done.counter += num_ready

            # Only keep the satellite points that aren't done yet
            keep_mask = ~ready_mask
            
            data_to_do.apply_mask(keep_mask)
        
