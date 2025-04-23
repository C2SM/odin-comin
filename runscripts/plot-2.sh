#!/bin/bash

cp wow.py ../simulation_outer_domain/
module load cray/23.12
module load cray-python/3.11.5
source venv-plot/bin/activate
cd ../simulation_outer_domain/ 

python wow.py