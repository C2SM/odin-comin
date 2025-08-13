#!/bin/bash

cp plot_station.py ../simulation_outer_domain_station/
module load cray/23.12
module load cray-python/3.11.5
source venv-plot/bin/activate
cd ../simulation_outer_domain_station/ 

python plot_station.py