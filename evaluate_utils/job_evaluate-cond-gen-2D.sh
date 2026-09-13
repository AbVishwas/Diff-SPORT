#!/usr/bin/env bash
# Scheduler header (SBATCH directives, cluster-specific environment setup) removed for the public release.
# Add your own scheduler directives / environment activation above the commands below if needed.

configname="OneObs2D_ds1_10M"
mask_type="from_ground_and_wall"

# Args: <mask%> <snaps> <num_plots>
mask=${1:-30}
snaps=${2:-100}    # number of snapshots to evaluate
num=${3:-3}        # number of instances to plot


echo $configname

python evaluate-cond-gen-2D.py ${configname} ${mask_type} ${mask} ${snaps} ${num} results/ 
