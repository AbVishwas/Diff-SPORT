#!/usr/bin/env bash
# Scheduler header (SBATCH directives, cluster-specific environment setup) removed for the public release.
# Add your own scheduler directives / environment activation above the commands below if needed.

configname="OneObs2D_ds1_10M" 
osp_method=$1
threshold=$2  #num_sensors for qr
start=$3
stop=$4
step=$5
datatype=$6
strategy=$7
seed=$8

python osp-cond-gen-pgdm-mapgd.py ${configname} ${osp_method} ${threshold} ${start} ${stop} ${step} ${datatype} ${strategy} ${seed}
