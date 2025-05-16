#!/bin/bash

cp test-str.py ../simulation_outer_domain/
module load cray/23.12
module load cray-python/3.11.5
source ../runscripts/venv-plot/bin/activate
cd ../simulation_outer_domain/ 

python test-str.py