#!/bin/bash

cp plot_flights_new.py ../simulation_outer_domain_flights/
module load cray/23.12
module load cray-python/3.11.5
source venv-plot/bin/activate
cd ../simulation_outer_domain_flights/ 

python plot_flights_new.py