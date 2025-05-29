#!/bin/bash

module load cray/23.12
module load cray-python/3.11.5
source venv-nc-gen/bin/activate

python convert_flights_to_nc.py