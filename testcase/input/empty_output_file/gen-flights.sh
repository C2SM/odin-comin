#!/bin/bash

module load cray/23.12
module load cray-python/3.11.5
source ../../plotting/venv-plot/bin/activate

python convert_flights_to_nc.py