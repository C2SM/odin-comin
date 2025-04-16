#!/bin/bash

export SPACK_SYSTEM_CONFIG_PATH=/user-environment/config/
export FI_CXI_OPTIMIZED_MRS=false

rm -rf slurm-*.out ../simulation_outer_domain

jobid=$(sbatch runscript_outer_write_restart_comin | awk '{print $4}')
squeue --me
sleep 2
tail -f slurm-${jobid}.out