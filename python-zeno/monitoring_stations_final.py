"""! @file monitoring_stations_final.py

@brief Tracking variables averaged over start and end times.

@section description_monitoring_stations_final Description
Tracking any variables averaged over starting and ending times.

@section libraries_monitoring_stations_final Libraries/Modules
- numpy
- xarray
- netCDF4
  - Access to Dataset
- pandas
  - Access to pandas.to_datetime
  - Access to pandas.read_csv
- sys

@section author_monitoring_stations_final Author(s)
- Created by Zeno Hug on 08/11/2025.

Copyright (c) 2025 Empa. All rights reserved.
"""

# Imports
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import pandas as pd
import sys

class StationDataToDoCIF:
    """!Holds station data for ComIn interface and provides masking capabilities.

    Stores spatial, temporal, and observational data for stations that still
    need processing, including interpolation metadata.

    Attributes:
        lon (float or np.ndarray): Longitude(s) of the station(s).
        lat (float or np.ndarray): Latitude(s) of the station(s).
        elevation (float or np.ndarray): Elevation above sea level (m).
        sampling_height (float or np.ndarray): Height above surface where samples are taken (m).
        sampling_strategy (int or np.ndarray): Strategy flag (1=lowland, 2=mountain, 3=instantaneous lowland, 4=instantaneous mountain).
        jc_loc (np.ndarray): JC grid indices.
        jb_loc (np.ndarray): JB grid indices.
        vertical_index1 (np.ndarray): First vertical interpolation index.
        vertical_index2 (np.ndarray): Second vertical interpolation index.
        vertical_weight (np.ndarray): Weights for vertical interpolation.
        horizontal_weight (np.ndarray): Weights for horizontal interpolation.
        number_of_steps (np.ndarray): Number of accumulation/averaging steps.
        id (str or np.ndarray): Station identifier(s).
        stime (np.datetime64 or np.ndarray): Start time(s).
        etime (np.datetime64 or np.ndarray): End time(s).
        parameter (str or np.ndarray): Parameter name(s) being measured.
        obs (np.ndarray): Accumulated observation values.
    """
    def __init__(self, lon, lat, elevation, sampling_height, sampling_strategy, jc_loc, jb_loc, vertical_index1, vertical_index2, vertical_weight, horizontal_weight, number_of_steps, id, stime, etime, parameter, obs):
        """!Initialize a StationDataToDoCIF instance.

        @param lon Longitude(s) of the station(s).
        @param lat Latitude(s) of the station(s).
        @param elevation Elevation above sea level (m).
        @param sampling_height Height above surface where samples are taken (m).
        @param sampling_strategy Strategy flag (1=lowland, 2=mountain, 3=instantaneous lowland, 4=instantaneous mountain).
        @param jc_loc JC grid indices.
        @param jb_loc JB grid indices.
        @param vertical_index1 First vertical interpolation index.
        @param vertical_index2 Second vertical interpolation index.
        @param vertical_weight Weights for vertical interpolation.
        @param horizontal_weight Weights for horizontal interpolation.
        @param number_of_steps Number of accumulation/averaging steps.
        @param id Station identifier(s).
        @param stime Start time(s).
        @param etime End time(s).
        @param parameter Parameter name(s) being measured.
        @param obs Accumulated observation values.
        """
        self.lon = lon
        self.lat = lat
        self.elevation = elevation
        self.sampling_height = sampling_height
        self.sampling_strategy = sampling_strategy
        self.jc_loc = jc_loc
        self.jb_loc = jb_loc
        self.vertical_index1 = vertical_index1
        self.vertical_index2 = vertical_index2
        self.vertical_weight = vertical_weight
        self.horizontal_weight = horizontal_weight
        self.number_of_steps = number_of_steps
        self.id = id
        self.stime = stime
        self.etime = etime
        self.parameter = parameter
        self.obs = obs

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes with matching length.

        Filters all attributes that are NumPy arrays whose first dimension
        equals ``mask.shape[0]``.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])

class StationDataToDo:
    """!Container for pending station computations across multiple variables.

    Stores station metadata plus per-variable accumulation arrays in
    ``dict_measurement`` for points that are not yet finished.

    Attributes:
        lon (np.ndarray): Longitudes of stations.
        lat (np.ndarray): Latitudes of stations.
        elevation (np.ndarray): Elevations above sea level (m).
        sampling_height (np.ndarray): Heights above surface for sampling (m).
        sampling_strategy (np.ndarray): Strategy flags (1,2,3,4).
        jc_loc (np.ndarray): JC grid indices (nearest neighbors).
        jb_loc (np.ndarray): JB grid indices (nearest neighbors).
        vertical_index1 (np.ndarray): First vertical interpolation indices.
        vertical_index2 (np.ndarray): Second vertical interpolation indices.
        vertical_weight (np.ndarray): Vertical interpolation weights.
        horizontal_weight (np.ndarray): Horizontal interpolation weights.
        number_of_steps (np.ndarray): Accumulated step counts per station.
        id (np.ndarray): Station identifiers.
        stime (np.ndarray): Start times.
        etime (np.ndarray): End times.
        dict_measurement (dict[str, np.ndarray]): Per-variable accumulation arrays.
    """
    def __init__(self, lon, lat, elevation, sampling_height, sampling_strategy, jc_loc, jb_loc, vertical_index1, vertical_index2, vertical_weight, horizontal_weight, number_of_steps, id, stime, etime, dict_vars):
        """!Initialize a StationDataToDo instance and allocate measurement arrays.

        @param lon Longitudes of stations.
        @param lat Latitudes of stations.
        @param elevation Elevations above sea level (m).
        @param sampling_height Heights above surface for sampling (m).
        @param sampling_strategy Strategy flags (1,2,3,4).
        @param jc_loc JC grid indices (nearest neighbors).
        @param jb_loc JB grid indices (nearest neighbors).
        @param vertical_index1 First vertical interpolation indices.
        @param vertical_index2 Second vertical interpolation indices.
        @param vertical_weight Vertical interpolation weights.
        @param horizontal_weight Horizontal interpolation weights.
        @param number_of_steps Accumulated step counts per station.
        @param id Station identifiers.
        @param stime Start times.
        @param etime End times.
        @param dict_vars Dictionary describing variables to compute; keys are variable
                         names used to allocate arrays in ``dict_measurement``.
        """
        self.lon = lon
        self.lat = lat
        self.elevation = elevation
        self.sampling_height = sampling_height
        self.sampling_strategy = sampling_strategy
        self.jc_loc = jc_loc
        self.jb_loc = jb_loc
        self.vertical_index1 = vertical_index1
        self.vertical_index2 = vertical_index2
        self.vertical_weight = vertical_weight
        self.horizontal_weight = horizontal_weight
        self.number_of_steps = number_of_steps
        self.id = id
        self.stime = stime
        self.etime = etime
        self.dict_measurement = {}
        for variable in dict_vars:
            self.dict_measurement[variable] = np.zeros(lon.shape, dtype=np.float64)

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes and per-variable arrays.

        Filters attributes that are NumPy arrays with the same first-dimension
        length as ``mask``, and applies the same mask to each array stored in
        ``dict_measurement``.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])
        # Also apply the mask to each variable in dict_measurement
        for variable, array in self.dict_measurement.items():
            if isinstance(array, np.ndarray) and array.shape[0] == mask.shape[0]:
                self.dict_measurement[variable] = array[mask]
    
class StationDataDoneCIF:
    """!Container for completed CIF station observations (ready to write out).

    Attributes:
        lon (np.ndarray): Longitudes of completed stations.
        lat (np.ndarray): Latitudes of completed stations.
        elevation (np.ndarray): Elevations above sea level (m).
        sampling_height (np.ndarray): Sampling heights above surface (m).
        sampling_strategy (np.ndarray): Strategy flags (1,2,3,4).
        stime (np.ndarray): Start times.
        etime (np.ndarray): End times.
        counter (int): Number of ready entries currently filled.
        id (np.ndarray): Station identifiers.
        parameter (np.ndarray): Parameter names.
        obs (np.ndarray): Final observation values.
    """
    def __init__(self, lon, lat, elevation, sampling_height, sampling_strategy, stime, etime, counter, id, parameter, obs):
        """!Initialize a StationDataDoneCIF instance.

        @param lon Longitudes.
        @param lat Latitudes.
        @param elevation Elevations above sea level (m).
        @param sampling_height Sampling heights above surface (m).
        @param sampling_strategy Strategy flags (1,2,3,4).
        @param stime Start times.
        @param etime End times.
        @param counter Number of filled entries so far.
        @param id Station identifiers.
        @param parameter Parameter names.
        @param obs Final observation values.
        """
        self.lon = lon
        self.lat = lat
        self.elevation = elevation
        self.sampling_height = sampling_height
        self.sampling_strategy = sampling_strategy
        self.stime = stime
        self.etime = etime
        self.counter = counter
        self.id = id
        self.parameter = parameter
        self.obs = obs

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes with matching length.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])

class StationDataDone:
    """!Container for completed multi-variable station results (ready to write out).

    Attributes:
        lon (np.ndarray): Longitudes of completed stations.
        lat (np.ndarray): Latitudes of completed stations.
        elevation (np.ndarray): Elevations above sea level (m).
        sampling_height (np.ndarray): Sampling heights above surface (m).
        sampling_strategy (np.ndarray): Strategy flags (1,2,3,4).
        stime (np.ndarray): Start times.
        etime (np.ndarray): End times.
        counter (int): Number of ready entries currently filled.
        id (np.ndarray): Station identifiers.
        dict_measurement (dict[str, np.ndarray]): Final per-variable values.
    """
    def __init__(self, lon, lat, elevation, sampling_height, sampling_strategy, stime, etime, counter, id, dict_vars):
        """!Initialize a StationDataDone instance and allocate output arrays.

        @param lon Longitudes.
        @param lat Latitudes.
        @param elevation Elevations above sea level (m).
        @param sampling_height Sampling heights above surface (m).
        @param sampling_strategy Strategy flags (1,2,3,4).
        @param stime Start times.
        @param etime End times.
        @param counter Number of filled entries so far.
        @param id Station identifiers.
        @param dict_vars Dictionary describing variables to store; keys are variable
                         names used to allocate arrays in ``dict_measurement``.
        """
        self.lon = lon
        self.lat = lat
        self.elevation = elevation
        self.sampling_height = sampling_height
        self.sampling_strategy = sampling_strategy
        self.stime = stime
        self.etime = etime
        self.counter = counter
        self.id = id
        self.dict_measurement = {}
        for variable in dict_vars:
            self.dict_measurement[variable] = np.zeros(lon.shape, dtype=np.float64)

    def apply_mask(self, mask):
        """!Apply a boolean mask to all array attributes with matching length.

        @param mask Boolean NumPy array indicating which elements to keep.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
                setattr(self, key, value[mask])


# Functions
def datetime64_to_days_since_1970(arr):
    """!Convert numpy datetime64 values to "days since 1970-01-01 00:00:00".

    @param arr Array-like or scalar of type ``np.datetime64`` to convert.
    @return Array-like or scalar of floats: days since the Unix epoch (UTC).
    """
    epoch = np.datetime64("1970-01-01T00:00:00", 'ns')
    return (arr - epoch) / np.timedelta64(1, 'D')

def lonlat2xyz(lon, lat):
    """!Convert spherical lon/lat (radians) to unit-sphere Cartesian coordinates.

    Expects angles in **radians**.

    @param lon Longitude(s) in radians.
    @param lat Latitude(s) in radians.
    @return Tuple ``(x, y, z)`` of arrays/scalars with Cartesian coordinates.
    """
    clat = np.cos(lat) 
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)

def find_points_cif(lons, lats, sampling_heights, sampling_elevations, sampling_strategies, tree, decomp_domain, clon, hhl, number_of_NN, ids, timesteps_begin, timesteps_end, accepted_distance, parameters):
    """!Locate CIF monitoring stations in the local PE domain and assemble interpolation metadata.

    All input lists/arrays must have the same length.

    @param lons Longitudes of target points (deg).
    @param lats Latitudes of target points (deg).
    @param sampling_heights Sampling heights above ground (m).
    @param sampling_elevations Ground elevations (m ASL).
    @param sampling_strategies Strategy flags: 1=lowland, 2=mountain, 3=instantaneous lowland, 4=instantaneous mountain.
    @param tree Nearest-neighbor search tree built on unit-sphere coordinates.
    @param decomp_domain Decomposition map indicating prognostic ownership per cell (0 = owned).
    @param clon Grid longitudes array (radians or degrees as used consistently downstream).
    @param hhl Half-level heights array (vertical column per (jc, jb)).
    @param number_of_NN Number of nearest cells to use for interpolation.
    @param ids Station identifiers (strings).
    @param timesteps_begin Start times per station.
    @param timesteps_end End times per station.
    @param accepted_distance Maximum accepted great-circle distance (km) to the closest cell.
    @param parameters Parameter names per station.
    @return Tuple of numpy arrays:
        - jc_locs (int32)         : JC indices (N x number_of_NN)
        - jb_locs (int32)         : JB indices (N x number_of_NN)
        - vertical_indices1 (int32): First vertical index (N x number_of_NN)
        - vertical_indices2 (int32): Second vertical index (N x number_of_NN)
        - weights_vertical_all (float64): Vertical weights (N x number_of_NN)
        - weights_all (float64)   : Horizontal weights (N x number_of_NN; rows sum to 1)
        - lons_local (float64)    : Accepted longitudes (N,)
        - lats_local (float64)    : Accepted latitudes (N,)
        - sampling_elevations_local (float64): Accepted elevations (N,)
        - sampling_heights_local (float64)   : Accepted sampling heights (N,)
        - sampling_strategy_local (int32)    : Accepted strategy flags (N,)
        - ids_local (U20)         : Accepted station IDs (N,)
        - timesteps_local_begin   : Accepted start times (N,)
        - timesteps_local_end     : Accepted end times (N,)
        - parameters_local (U20)  : Accepted parameters (N,)
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
    parameters_local = []

    # Loop through every station
    for lon, lat, sampling_height, sampling_elevation, sampling_strategy, id, timestep_begin, timestep_end, parameter in zip(lons, lats, sampling_heights, sampling_elevations, sampling_strategies, ids, timesteps_begin, timesteps_end, parameters):

        # Query the tree for the NUMBER_OF_NN nearest cells
        dd, ii = tree.query([lonlat2xyz(np.deg2rad(lon), np.deg2rad(lat))], k = number_of_NN)

        closest_distance = dd[0][0] * 6371.0

        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0][0]] == 0 and closest_distance <= accepted_distance:
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
            parameters_local.append(parameter)

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
            np.array(timesteps_local_end), 
            np.array(parameters_local, dtype = 'U20'))


def find_points(lons, lats, sampling_heights, sampling_elevations, sampling_strategies, tree, decomp_domain, clon, hhl, number_of_NN, ids, timesteps_begin, timesteps_end, accepted_distance):
    """!Locate stationary monitoring stations in the local PE domain and assemble interpolation metadata.

    All input lists/arrays must have the same length.

    @param lons Longitudes of target points (deg).
    @param lats Latitudes of target points (deg).
    @param sampling_heights Sampling heights above ground (m).
    @param sampling_elevations Ground elevations (m ASL).
    @param sampling_strategies Strategy flags: 1=lowland, 2=mountain, 3=instantaneous lowland, 4=instantaneous mountain.
    @param tree Nearest-neighbor search tree built on unit-sphere coordinates.
    @param decomp_domain Decomposition map indicating prognostic ownership per cell (0 = owned).
    @param clon Grid longitudes array.
    @param hhl Half-level heights array.
    @param number_of_NN Number of nearest cells to use for interpolation.
    @param ids Station identifiers (strings).
    @param timesteps_begin Start times per station.
    @param timesteps_end End times per station.
    @param accepted_distance Maximum accepted great-circle distance (km) to the closest cell.
    @return Tuple of numpy arrays like ``find_points_cif`` but without ``parameters``.
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

        closest_distance = dd[0][0] * 6371.0

        # Check if the nearest cell is in this PE's domain and is owned by this PE. This ensures that each station is only done by one PE
        if decomp_domain.ravel()[ii[0][0]] == 0 and closest_distance <= accepted_distance:
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

def write_points_cif(comm, data_done: StationDataDoneCIF, file_name_output):
    """!Write completed CIF station observations to a NetCDF file (append mode).

    Uses preallocated arrays and a counter to append new observations collected
    across ranks. Root (rank 0) concatenates, sorts by ``etime``, converts times
    to "days since 1970-01-01", and writes variables.

    @param comm MPI communicator.
    @param data_done Completed data container (``StationDataDoneCIF``).
    @param file_name_output Path to an existing NetCDF file with header already written.
    @return None.
    """
    done_counter = data_done.counter
    done_data_local = None
    if done_counter > 0:

        done_data_local = {
            "site_name": data_done.id[:done_counter],
            "longitude": data_done.lon[:done_counter],
            "latitude": data_done.lat[:done_counter],
            "elevation": data_done.elevation[:done_counter],
            "sampling_height": data_done.sampling_height[:done_counter],
            "sampling_strategy": data_done.sampling_strategy[:done_counter],
            "stime": data_done.stime[:done_counter],
            "etime": data_done.etime[:done_counter],
            "parameter": data_done.parameter[:done_counter],
            "observation": data_done.obs[:done_counter],
        }

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
            "parameter": [],
            "observation": [],
        }

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
            

    data_done.counter = 0

def write_points(comm, data_done: StationDataDone, dict_vars, file_name_output):
    """!Write completed multi-variable station data to a NetCDF file (append mode).

    Root (rank 0) gathers from all ranks, concatenates, sorts by ``etime``, converts
    times to "days since 1970-01-01", then writes core metadata and each variable
    in ``dict_vars``.

    @param comm MPI communicator.
    @param data_done Completed data container (``StationDataDone``).
    @param dict_vars Dictionary describing variables (units, long_name, etc.).
    @param file_name_output Path to an existing NetCDF file with header already written.
    @return None.
    """
    done_counter = data_done.counter
    done_data_local = None
    if done_counter > 0:

        done_data_local = {
            "site_name": data_done.id[:done_counter],
            "longitude": data_done.lon[:done_counter],
            "latitude": data_done.lat[:done_counter],
            "elevation": data_done.elevation[:done_counter],
            "sampling_height": data_done.sampling_height[:done_counter],
            "sampling_strategy": data_done.sampling_strategy[:done_counter],
            "stime": data_done.stime[:done_counter],
            "etime": data_done.etime[:done_counter],
        }

        for variable in dict_vars:
            done_data_local[variable] = data_done.dict_measurement[variable][:done_counter]

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
            

    data_done.counter = 0

def write_header_points_cif(comm, file_name):
    """!Create the NetCDF header for CIF monitoring output.

    Defines dimensions and variables, assigns CF-compliant metadata and
    global attributes. Only executed on root (rank 0).

    @param comm MPI communicator.
    @param file_name Output NetCDF file path to create (will overwrite).
    @return None.
    """
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
        sampling_strategy.comment = "1=low ; 2=mountain ; 3=lowland instantaneous measurement; 4=mountain instantaneous measurement"

        site_name = ncfile.createVariable('site_name', 'S1', ('obs', 'nchar'))
        site_name.long_name = "station name or ID"

        parameter = ncfile.createVariable('parameter', 'S10', ('obs',))
        parameter.long_name = "The variable that the observation depicts"

        observation = ncfile.createVariable('observation', 'f4', ('obs',), fill_value=1.0e+20)
        observation.long_name = "The actual observation. See parameter for what was measured here"

        # Global attributes
        ncfile.Conventions = "CF-1.8"
        ncfile.title = "Station output file for ICON ComIn interface XYZ"
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



def write_header_points(comm, file_name, dict_vars):
    """!Create the NetCDF header for stationary monitoring output and add user variables.

    Defines base dimensions/variables and CF metadata on root (rank 0),
    then appends one variable per entry in ``dict_vars`` (with ``unit`` and ``long_name``).

    @param comm MPI communicator.
    @param file_name Output NetCDF file path to create (will overwrite/append).
    @param dict_vars Mapping ``{var_name: {'unit': str, 'long_name': str}}``.
    @return None.
    """
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
        sampling_strategy.comment = "1=low ; 2=mountain ; 3=lowland instantaneous measurement; 4=mountain instantaneous measurement"

        site_name = ncfile.createVariable('site_name', 'S1', ('obs', 'nchar'))
        site_name.long_name = "station name or ID"

        # Global attributes
        ncfile.Conventions = "CF-1.8"
        ncfile.title = "Station output file for ICON ComIn interface XYZ"
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


def read_in_points(comm, tree, decomp_domain, clon, hhl, number_of_NN, path_to_file, start_model, end_model, data_vars, accepted_distance):
    """!Read stationary monitoring configuration from NetCDF and build to-do/done containers.

    Filters stations to the model time window, broadcasts inputs to all ranks,
    locates stations in the local domain via ``find_points``, and allocates
    ``StationDataToDo`` / ``StationDataDone`` structures.

    @param comm MPI communicator.
    @param tree Nearest-neighbor search tree (unit-sphere).
    @param decomp_domain Decomposition/ownership map per cell.
    @param clon Grid longitudes array.
    @param hhl Half-level heights array.
    @param number_of_NN Number of nearest neighbors for interpolation.
    @param path_to_file Path to input NetCDF with station definitions.
    @param start_model Experiment start datetime.
    @param end_model Experiment end datetime.
    @param data_vars Dict describing variables to compute (used for allocation).
    @param accepted_distance Maximum accepted distance (km) to closest cell.
    @return Tuple ``(data_to_do, data_done)`` where each is a class instance.
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
        lons, lats, sampling_heights, elevations, sampling_strategies, tree, decomp_domain, clon, hhl, number_of_NN, site_names, stimes, etimes, accepted_distance)

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

    number_of_timesteps = np.zeros(N_points, dtype=np.int32)
    data_to_do = StationDataToDo(lons, lats, elevations, sampling_heights, sampling_strategies, jc_loc, jb_loc, vertical_indices_nearest, vertical_indices_second, vertical_weights, horizontal_weights, number_of_timesteps, ids, stimes, etimes, data_vars)
    data_done = StationDataDone(done_lons, done_lats, done_elevations, done_sampling_heights, done_sampling_strategies, done_stimes, done_etimes, done_counter, done_site_names, data_vars)

    return data_to_do, data_done # Return the data


def read_in_points_cif(comm, tree, decomp_domain, clon, hhl, number_of_NN, path_to_file, start_model, end_model, data_vars, accepted_distance):
    """!Read CIF monitoring configuration from CSV and build to-do/done containers.

    Loads station rows, converts times, filters to the model window, broadcasts to all
    ranks, locates stations via ``find_points_cif``, and allocates
    ``StationDataToDoCIF`` / ``StationDataDoneCIF`` structures.

    @param comm MPI communicator.
    @param tree Nearest-neighbor search tree (unit-sphere).
    @param decomp_domain Decomposition/ownership map per cell.
    @param clon Grid longitudes array.
    @param hhl Half-level heights array.
    @param number_of_NN Number of nearest neighbors for interpolation.
    @param path_to_file Path to CSV input.
    @param start_model Experiment start datetime.
    @param end_model Experiment end datetime.
    @param data_vars Dict describing variables to compute (used for allocation).
    @param accepted_distance Maximum accepted distance (km) to closest cell.
    @return Tuple ``(data_to_do, data_done)`` where each is a class instance.
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
        sampling_strategies = df['flags'].to_numpy()
        site_names = df['station']
        parameters = df['parameter']
        
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
        parameters = parameters[valid_mask]
    else:
        lons = None
        lats = None
        elevations = None
        sampling_heights = None
        sampling_strategies = None
        site_names = None
        stimes = None
        etimes = None
        parameters = None

    
    # Broadcast the data to all processes, from root 0
    lons = comm.bcast(lons, root = 0)
    lats = comm.bcast(lats, root = 0)
    elevations = comm.bcast(elevations, root = 0)
    sampling_heights = comm.bcast(sampling_heights, root = 0)
    sampling_strategies = comm.bcast(sampling_strategies, root = 0)
    site_names = comm.bcast(site_names, root = 0)
    stimes = comm.bcast(stimes, root = 0)
    etimes = comm.bcast(etimes, root = 0)
    parameters = comm.bcast(parameters, root = 0)

     # Find all of the monitoring stations in this local PE's domain and save all relevant data
    (jc_loc, jb_loc, vertical_indices_nearest, vertical_indices_second, vertical_weights,  horizontal_weights, 
        lons, lats, elevations, sampling_heights, sampling_strategies, ids, stimes, etimes, parameters) = find_points_cif(
        lons, lats, sampling_heights, elevations, sampling_strategies, tree, decomp_domain, clon, hhl, number_of_NN, site_names, stimes, etimes, accepted_distance, parameters)

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
    done_parameters = np.empty(N_points, dtype='U20')
    done_obs = np.empty(N_points, dtype=np.float64)
    obs = np.zeros(N_points, dtype=np.float64)

    done_counter = 0 # counter of how many of the points are already done (since last writeout)

    # Create Dicts with all of the data needed
    number_of_timesteps = np.zeros(N_points, dtype=np.int32)
    data_to_do = StationDataToDoCIF(lons, lats, elevations, sampling_heights, sampling_strategies, jc_loc, jb_loc, vertical_indices_nearest, vertical_indices_second, vertical_weights, horizontal_weights, number_of_timesteps, ids, stimes, etimes, parameters, obs)
    data_done = StationDataDoneCIF(done_lons, done_lats, done_elevations, done_sampling_heights, done_sampling_strategies, done_stimes, done_etimes, done_counter, done_site_names, done_parameters, done_obs)

    return data_to_do, data_done # Return the dicts with the data

def tracking_points(datetime, data_to_do: StationDataToDo, data_done: StationDataDone, data_np, dict_vars, operations_dict):
    """!Accumulate and finalize multi-variable station measurements for the current model time.

    For stations whose time window includes the current model time, perform vertical and
    horizontal interpolation, accumulate results in ``data_to_do.dict_measurement``,
    and, when finished (``etime`` passed), move averaged values to ``data_done``.

    @param datetime Current model datetime (``np.datetime64``-compatible).
    @param data_to_do Pending data container (``StationDataToDo``).
    @param data_done Completed data container (``StationDataDone``).
    @param data_np Mapping ``{var_name: list[np.ndarray]}`` of model fields involved in a formula.
    @param dict_vars Variable recipe dict with ``factor`` and ``signs`` lists per variable.
    @param operations_dict Mapping from sign string to binary operator (e.g., ``np.add``, ``np.subtract``).
    @return None.
    """
    if data_to_do.lon.size > 0: # Checks if there is still work to do
        
        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is in the hour before the output of the measurement. They are ready for measurement
        measuring_mask = (
            (((data_to_do.sampling_strategy == 1) | (data_to_do.sampling_strategy == 2)) &
            (data_to_do.stime <= model_time_np) & 
            (data_to_do.etime >= model_time_np)) |
            (((data_to_do.sampling_strategy == 3) | (data_to_do.sampling_strategy == 4)) &
            (data_to_do.etime <= model_time_np))
        )
        if np.any(measuring_mask):
            # Filter arrays for ready stations
            jc_ready = data_to_do.jc_loc[measuring_mask]
            jb_ready = data_to_do.jb_loc[measuring_mask]
            vi_ready1 = data_to_do.vertical_index1[measuring_mask]
            vi_ready2 = data_to_do.vertical_index2[measuring_mask]
            weights_vertical_ready = data_to_do.vertical_weight[measuring_mask]
            weights_ready = data_to_do.horizontal_weight[measuring_mask]
            
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
                    data_to_do.dict_measurement[variable][measuring_mask] += np.sum(weights_ready * monitoring_combined, axis=1)
            
            data_to_do.number_of_steps[measuring_mask] += 1


        done_mask = data_to_do.etime <= model_time_np # This data is done being monitored and can be output
        done_counter = data_done.counter
        num_ready = np.sum(done_mask) # Count how many points are done
        if num_ready > 0:
        # Add all of the done points to the done arrays
            data_done.lon[done_counter:done_counter + num_ready] = data_to_do.lon[done_mask]
            data_done.lat[done_counter:done_counter + num_ready] = data_to_do.lat[done_mask]
            data_done.elevation[done_counter:done_counter + num_ready] = data_to_do.elevation[done_mask]
            data_done.sampling_height[done_counter:done_counter + num_ready] = data_to_do.sampling_height[done_mask]
            data_done.sampling_strategy[done_counter:done_counter + num_ready] = data_to_do.sampling_strategy[done_mask]
            data_done.stime[done_counter:done_counter + num_ready] = data_to_do.stime[done_mask]
            data_done.etime[done_counter:done_counter + num_ready] = data_to_do.etime[done_mask]
            data_done.id[done_counter:done_counter + num_ready] = data_to_do.id[done_mask]

            # Averaging of the data, as before we just added up contributions
            # Current behaviour is that if there are no steps made, I divide by 0 which results in NaN, which is actually great as I then in post processing know for which points I dont have data
            for variable in dict_vars:
                data_done.dict_measurement[variable][done_counter:done_counter + num_ready] = data_to_do.dict_measurement[variable][done_mask] / data_to_do.number_of_steps[done_mask]


            # Keep count of how many points are done
            data_done.counter+= num_ready

            # Only keep the points that aren't done yet
            keep_mask = ~done_mask

            data_to_do.apply_mask(keep_mask)


def tracking_points_cif(datetime, data_to_do: StationDataToDoCIF, data_done: StationDataDoneCIF, data_np, dict_vars, operations_dict):
    """!Accumulate and finalize CIF station observations for the current model time.

    For each ready station/parameter, evaluate the configured expression using
    vertical and horizontal interpolation, accumulate into ``data_to_do.obs``,
    and move finished, averaged values to ``data_done``.

    @param datetime Current model datetime (``np.datetime64``-compatible).
    @param data_to_do Pending data container (``StationDataToDoCIF``).
    @param data_done Completed data container (``StationDataDoneCIF``).
    @param data_np Mapping ``{parameter: list[np.ndarray]}`` of model fields.
    @param dict_vars Recipe dict per parameter with ``factor`` and ``signs``.
    @param operations_dict Mapping from sign string to binary operator (e.g., ``np.add``).
    @return None.
    """
    if data_to_do.lon.size > 0: # Checks if there is still work to do
        
        model_time_np = np.datetime64(datetime)
        # mask to mask out the stations, where the model time is in the hour before the output of the measurement. They are ready for measurement
        measuring_mask = (
            (((data_to_do.sampling_strategy == 1) | (data_to_do.sampling_strategy == 2)) &
            (data_to_do.stime <= model_time_np) & 
            (data_to_do.etime >= model_time_np)) |
            (((data_to_do.sampling_strategy == 3) | (data_to_do.sampling_strategy == 4)) &
            (data_to_do.etime <= model_time_np))
        )
        if np.any(measuring_mask):
            # Filter arrays for ready stations
            jc_ready = data_to_do.jc_loc[measuring_mask]
            jb_ready = data_to_do.jb_loc[measuring_mask]
            vi_ready1 = data_to_do.vertical_index1[measuring_mask]
            vi_ready2 = data_to_do.vertical_index2[measuring_mask]
            weights_vertical_ready = data_to_do.vertical_weight[measuring_mask]
            weights_ready = data_to_do.horizontal_weight[measuring_mask]
            parameters_ready = data_to_do.parameter[measuring_mask]
            
            indices = np.where(measuring_mask)[0]

            for jc, jb, vi1, vi2, weight_vertical, weights_horizontal, parameter, index in zip(jc_ready, jb_ready, vi_ready1, vi_ready2, weights_vertical_ready, weights_ready, parameters_ready, indices):
                list_data = data_np[parameter]
                monitoring_1 = list_data[0][jc, vi1, jb, 0, 0] * dict_vars[parameter]['factor'][0]
                monitoring_2 = list_data[0][jc, vi2, jb, 0, 0] * dict_vars[parameter]['factor'][0]
                for sign, i in zip(dict_vars[parameter]['signs'], range(1, len(list_data))):
                    monitoring_1 = operations_dict[sign](monitoring_1, list_data[i][jc, vi1, jb, 0, 0] * dict_vars[parameter]['factor'][i])
                    monitoring_2 = operations_dict[sign](monitoring_2, list_data[i][jc, vi2, jb, 0, 0] * dict_vars[parameter]['factor'][i])
                # Do the vertical interpolation
                monitoring_combined = monitoring_1 + weight_vertical * (monitoring_2 - monitoring_1)
                # Do the horizontal interpolation
                if weights_horizontal.size > 0 and monitoring_combined.size > 0:
                    data_to_do.obs[index] += np.sum(weights_horizontal * monitoring_combined)
                data_to_do.number_of_steps[index] +=1


        done_mask = data_to_do.etime <= model_time_np # This data is done being monitored and can be output
        done_counter = data_done.counter
        num_ready = np.sum(done_mask) # Count how many points are done
        if num_ready > 0:
        # Add all of the done points to the done arrays
            data_done.lon[done_counter:done_counter + num_ready] = data_to_do.lon[done_mask]
            data_done.lat[done_counter:done_counter + num_ready] = data_to_do.lat[done_mask]
            data_done.elevation[done_counter:done_counter + num_ready] = data_to_do.elevation[done_mask]
            data_done.sampling_height[done_counter:done_counter + num_ready] = data_to_do.sampling_height[done_mask]
            data_done.sampling_strategy[done_counter:done_counter + num_ready] = data_to_do.sampling_strategy[done_mask]
            data_done.stime[done_counter:done_counter + num_ready] = data_to_do.stime[done_mask]
            data_done.etime[done_counter:done_counter + num_ready] = data_to_do.etime[done_mask]
            data_done.id[done_counter:done_counter + num_ready] = data_to_do.id[done_mask]
            data_done.parameter[done_counter:done_counter + num_ready] = data_to_do.parameter[done_mask]
            data_done.obs[done_counter:done_counter + num_ready] = data_to_do.obs[done_mask] / data_to_do.number_of_steps[done_mask]



            # Keep count of how many points are done
            data_done.counter += num_ready

            # Only keep the points that aren't done yet
            keep_mask = ~done_mask

            data_to_do.apply_mask(keep_mask)
        
