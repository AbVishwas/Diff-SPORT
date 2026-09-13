#!/usr/bin/env bash
# Scheduler header (SBATCH directives, cluster-specific environment setup) removed for the public release.
# Add your own scheduler directives / environment activation above the commands below if needed.

set -euo pipefail

configname="OneObs2D_ds1_10M"
mask_type="from_ground_and_wall"
mask=${1:-30}
snaps=1200
gen_seed=${2:-0}
mask_seed=${3:-0}
select_random_sensors=${4:-}


if [[ -n "$select_random_sensors" ]]; then
  python ddrm-pigdm-mapgd-gen-2D.py ${configname} ${mask_type} ${mask} ${snaps} ${gen_seed} ${mask_seed} ${select_random_sensors}
else
  python ddrm-pigdm-mapgd-gen-2D.py ${configname} ${mask_type} ${mask} ${snaps} ${gen_seed} ${mask_seed}
fi

