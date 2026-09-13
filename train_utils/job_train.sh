#!/usr/bin/env bash
# Scheduler header (SBATCH directives, cluster-specific environment setup) removed for the public release.
# Add your own scheduler directives / environment activation above the commands below if needed.

configname=$1
python train.py ${configname} 
#python resume.py ${configname} 
