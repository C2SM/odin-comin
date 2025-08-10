#!/bin/bash

module load cray/23.12
module load cray-python/3.11.5
source ../../plotting/venv-plot/bin/activate

python generate_station_file.py