#!/usr/bin/env bash
set -euo pipefail

# Orchestrates the full diffSPORT workflow for Code Ocean
# Usage: ./run_co.sh [--skip-train]

SKIP_TRAIN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$ROOT_DIR/run.sh"
CONFIG="OneObs2D_ds1_10M"
MASK_TYPE="from_ground_and_wall"
MASK=30

# 1. Train diffusion model
if [[ $SKIP_TRAIN -eq 0 ]]; then
  bash "$RUN_SCRIPT" train --config "$CONFIG"
else
  echo "[run_co] Skipping training step (--skip-train set)."
fi

# 2. Generate unconditional samples for multiple seeds
bash "$RUN_SCRIPT" gen-uncond --config "$CONFIG" --seeds 1,2,3,4,5 --out inference_utils/unconditional_generation/generated_samples

# 3. Pack unconditional trajectories into a single HDF5
bash "$RUN_SCRIPT" pack-uncond --config "$CONFIG" --dir inference_utils/unconditional_generation/generated_samples

# 4. Evaluate unconditional reconstructions
bash "$RUN_SCRIPT" eval-uncond --config "$CONFIG" --num 3 --results results/

# 5. Generate baseline conditional samples
bash "$RUN_SCRIPT" gen-cond --config "$CONFIG" --mask_type "$MASK_TYPE" --mask "$MASK" --snaps 25000 --gen_seed 0 --mask_seed 0

# 6. Evaluate baseline conditional generation
bash "$RUN_SCRIPT" eval-cond --config "$CONFIG" --mask_type "$MASK_TYPE" --mask "$MASK" --snaps 25000 --num 20 --results results/

# 7. Compute SHAP values for training set segment
bash "$RUN_SCRIPT" shap-values --config "$CONFIG" --mask_type "$MASK_TYPE" --mask "$MASK" --shap_start 0 --shap_stop 25000 --shap_step 50 --shap_ncoalitions 3000 --shap_select sequential

echo "[run_co] Manual step required: run QR pivoting notebook (osp_utils/qr-pivoting/qr-pivoting-v4.ipynb) to export qr_mask NPZ files."
echo "[run_co] Manual step required: run SHAP aggregation notebook (osp_utils/shap/related_notebooks/v5.5-tdiff20-gd50-snaps-25000-step-50.ipynb) to export threshold NPZ files."

echo "[run_co] After manual notebook steps complete, continue with steps 10-11 manually:"
echo "  ./run.sh gen-osp --config $CONFIG --method qr --seeds 0,1,2,3,4,5 --start 0 --stop 25000 --step 50"
echo "  ./run.sh gen-osp --config $CONFIG --method shap --seeds 0,1,2,3,4,5 --start 0 --stop 25000 --step 50"
echo "  ./run.sh gen-cond --config $CONFIG --mask_type $MASK_TYPE --mask $MASK --snaps 25000 --gen_seed 0 --mask_seed 0 --random_sensors 120"
echo "  ./run.sh eval-best --config $CONFIG"
