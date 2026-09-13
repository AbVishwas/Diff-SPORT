#!/usr/bin/env bash
# Scheduler header (SBATCH directives, cluster-specific environment setup) removed for the public release.
# Add your own scheduler directives / environment activation above the commands below if needed.

seed=$1
configname=$2
#savedir=$3

python generate_samples.py ${seed} ${configname} .
