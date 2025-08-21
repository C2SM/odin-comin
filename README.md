# ODIN-ComIn: Online Data Interpolator for ICON-ART

ODIN (Online Data INterpolator) is a plugin for the [ICON-ART atmospheric transport model](https://mpimet.mpg.de/en/science/models/icon-esm/icon-art) that allows direct sampling of model variables at user-defined observation locations **during runtime**. It is implemented using the [ComIn interface](https://comin.readthedocs.io/) and written in Python.

The plugin enables efficient and reproducible extraction of model output for:
- **Monitoring stations** (time-averaged or instantaneous)
- **Mobile trajectories** (aircraft, drones, vehicles)
- **Satellite retrievals** (e.g. TROPOMI CH₄ with averaging kernels and stratospheric extension)

Key features:
- Horizontal and vertical interpolation at arbitrary locations
- Application of averaging kernels and CAMS-based profile extension
- Output directly to structured NetCDF files
- Fully configured via a YAML file (no code changes required)
- Parallel scaling with MPI

---

## Installation & Compilation

To use ODIN within ICON:

1. **Compile ICON with ComIn support enabled.**
   - Make sure the `--enable-comin` option is active in the ICON build system.

2. **Build the ComIn Python adapter.**
   - Provides efficient NumPy-based access to ICON arrays.

3. **Clone this repository** into your working environment.

4. **Install Python dependencies**:

Detailed build notes for CSCS Eiger are provided in docs/notes.txt.


## Configuration
All runtime options are handled via a YAML configuration file, e.g.:

NUMBER_OF_NN: 4
time_interval_writeout: 3600
accepted_distance: 12.0
jg: 1
msgrank: 0

dict_vars:
  CH4:
    var_names: ["CH4_EMIS", "CH4_BG"]
    signs: ["plus"]
    factor: [1.0e9, 1.0]
    unit: "ppb"
    long_name: "CH4 concentration"
  Temp:
    var_names: ["temp"]
    signs: []
    factor: [1]
    unit: "Kelvin"
    long_name: "Temperature"
  Temp:
    var_names: ["temp"]
    signs: []
    factor: [1]
    unit: "Kelvin"
    long_name: "Temperature"

dict_vars_cif_sat:
  CH4:
    var_names: ["CH4_EMIS", "CH4_BG"]
    signs: ["plus"]
    factor: [1.0e9, 1.0]
    unit: "ppb"
    long_name: "CH4 concentration"
  Temp:
    var_names: ["temp"]
    signs: []
    factor: [1]
    unit: "Kelvin"
    long_name: "Temperature"

dict_vars_cif_stations:
  CH4:
    var_names: ["CH4_EMIS", "CH4_BG"]
    signs: ["plus"]
    factor: [1.0e9, 1.0]
    unit: "ppb"
    long_name: "CH4 concentration"
  Temp:
    var_names: ["temp"]
    signs: []
    factor: [1]
    unit: "Kelvin"
    long_name: "Temperature"

do_monitoring_stations: false
do_satellite_CH4: true
do_satellite_cif: false
do_stations_cif: false

plugin_dir: "/capstor/scratch/cscs/zhug/Romania6km/plugin"
tropomi_filename: "/capstor/scratch/cscs/zhug/Romania6km/testcase/input/TROPOMI/TROPOMI_SRON_corners_20190101_20191231.nc"
cams_base_path: "/capstor/scratch/cscs/zhug/Romania6km/testcase/input/CAMS/LBC/"
cams_params_file: "/capstor/scratch/cscs/zhug/Romania6km/testcase/input/CAMS/cams_params_minimal.nc"
path_to_input_nc: "/capstor/scratch/cscs/zhug/Romania6km/testcase/input/empty_output_file/input_flight.nc"
path_to_input_sat_cif: "/capstor/scratch/cscs/zhug/Romania6km/testcase/input/cif/sat.csv"
path_to_input_stations_cif: "/capstor/scratch/cscs/zhug/Romania6km/testcase/input/cif/surf.csv"
file_name_output_sat_cif: "output_sat_cif.nc"
file_name_output_stations_cif: "output_stations_cif.nc"
file_name_output: "output.nc"
file_name_output_sat_CH4: "output_sat_ch4.nc"

## Usage

Run ICON with ODIN by adding the plugin to the runscript.

## Testcase
A minimal testcase is provided in this repository under testcase/:
	•	ICON domain over Romania at 6 km resolution
	•	Predefined monitoring stations and satellite inputs
	•	Example YAML configuration
	•	Instructions for compiling and running
Run the testcase to verify your setup and reproduce the workflows described in the thesis.

## Example Applications

	•	Monitoring stations: Extract CH₄ and temperature at ICOS sites with hourly averaging.
	•	Aircraft flights: Sample ICON along measured flight tracks at model timestep resolution.
	•	Satellite retrievals: Generate TROPOMI-equivalent XCH₄ with averaging kernels and CAMS extension.

## Data & Code Availability

The ODIN-ComIn plugin and testcase data are hosted here:
https://gitlab.com/empa503/atmospheric-modelling/odin-comin.git 

The Bachelor Thesis describing the methods in detail:
ODIN: An online data interpolator for the ICON-ART atmospheric transport model using the ComIn interface (Zeno Hug, 2025)

⸻

License

This project is released under the MIT License.
See LICENSE for details.

⸻

Citation

If you use this software in scientific work, please cite:

Zeno Hug (2025): ODIN: An online data interpolator for the ICON-ART atmospheric transport model using the ComIn interface. Bachelor Thesis, Empa, Atmospheric Modelling and Remote Sensing group.