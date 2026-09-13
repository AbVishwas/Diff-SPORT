#!/usr/bin/env bash
# Scheduler header (SBATCH directives, cluster-specific environment setup) removed for the public release.
# Add your own scheduler directives / environment activation above the commands below if needed.

# Input file with parameter sets
input_file="input_params.txt"

# Read and evaluate parameter line based on SLURM_ARRAY_TASK_ID
param_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$input_file")
eval "$param_line"

# Static config
configname="OneObs2D_ds1_10M" 
mask_type="from_ground_and_wall"
mask=30
select_data="sequential"  # or "random"

echo "[`date`] Starting job $SLURM_ARRAY_TASK_ID with: start=$start, stop=$stop, step=$step, ncoalitions=$coalition"

# Run your Python script with the parsed arguments
python obs2D-osp-shap.py "$configname" "$mask_type" "$mask" "$start" "$stop" "$step" "$coalition" "$select_data"

echo "[`date`] Finished job $SLURM_ARRAY_TASK_ID"
