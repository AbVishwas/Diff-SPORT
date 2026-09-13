#!/usr/bin/env bash
# Diff-SPORT runner: one entry point for training, sampling, sensor placement and evaluation.
#
# Usage: ./run.sh <command> [--option value ...]
#
# Commands
#   train         Train the diffusion prior for a config.
#   gen-uncond    Generate unconditional samples for one or more seeds.
#   pack-uncond   Pack per-seed unconditional samples into a single HDF5 file.
#   eval-uncond   Statistical evaluation of unconditional samples (Reynolds stresses, PDFs, PSDs).
#   gen-cond      Conditional reconstruction (MAPGD, PiGDM, DDRM) from a sensor mask.
#   eval-cond     Instantaneous and error-PDF evaluation of conditional reconstructions.
#   shap-values   Compute SHAP values of sensor subregions (batched KernelSHAP, modified kernel).
#   gen-osp       Reconstruction sweeps over QR-pivoting or SHAP sensor masks at several thresholds.
#   eval-best     Compare SHAP, QR and random placement (error vs. sensor coverage).
#   shap-summary  Re-plot SHAP/QR comparisons from cached results.
#
# Options: --config, --mask_type, --mask, --snaps, --num, --gen_seed, --mask_seed, --random_sensors,
#   --results, --out/--dir, --seeds, --method qr|shap, --start, --stop, --step, --strategy,
#   --qr_thresholds, --shap_thresholds, --shap_start, --shap_stop, --shap_step, --shap_ncoalitions,
#   --shap_select. Every option has a default set in the "Defaults" block below.
#
# The defaults are small smoke-test values (few snapshots, few coalitions) so that each command
# runs quickly. For the settings used in the paper, see the README; pass them as options.
#
# Manual notebook steps between commands:
#   osp_utils/qr-pivoting/qr-pivoting-v4.ipynb                                     derive QR-pivoting masks
#   osp_utils/shap/related_notebooks/v5.5-tdiff20-gditer50-snaps-25000-step-50.ipynb   aggregate SHAP values into threshold masks

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Defaults
CONFIG="OneObs2D_ds1_10M"
MASK_TYPE="from_ground_and_wall"
MASK=30
SNAPS=100
NUM=3
GEN_SEED=0
MASK_SEED=0
RANDOM_SENSORS=""
RESULTS_DIR="results"
UNCOND_OUT_DIR="inference_utils/unconditional_generation/generated_samples"
SEEDS="1,2,3,4,5"
OSP_METHOD="shap"  # qr|shap
START=0
STOP=1000
STEP=50
STRATEGY="mean"    # used for shap

QR_THRESHOLDS="22.0,38.0,71.0"
SHAP_THRESHOLDS="0.73,0.74,0.75,0.76,0.77,0.775,0.78,0.8,0.82,0.84,0.86,0.9"

SHAP_START=0
SHAP_STOP=1000
SHAP_STEP=50
SHAP_NCOALITIONS=100
SHAP_SELECT_DATA="sequential"

usage() {
  # print the comment header (everything after the shebang up to the first non-comment line)
  awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
}

parse_kv() {
  for ((i=1; i<=$#; i++)); do
    case "${!i}" in
      --config) ((i++)); CONFIG="${!i}" ;;
      --mask_type) ((i++)); MASK_TYPE="${!i}" ;;
      --mask) ((i++)); MASK="${!i}" ;;
      --snaps) ((i++)); SNAPS="${!i}" ;;
      --num) ((i++)); NUM="${!i}" ;;
      --gen_seed) ((i++)); GEN_SEED="${!i}" ;;
      --mask_seed) ((i++)); MASK_SEED="${!i}" ;;
      --random_sensors) ((i++)); RANDOM_SENSORS="${!i}" ;;
      --results) ((i++)); RESULTS_DIR="${!i}" ;;
      --dir|--out) ((i++)); UNCOND_OUT_DIR="${!i}" ;;
      --seeds) ((i++)); SEEDS="${!i}" ;;
      --method) ((i++)); OSP_METHOD="${!i}" ;;
      --start) ((i++)); START="${!i}" ;;
      --stop) ((i++)); STOP="${!i}" ;;
      --step) ((i++)); STEP="${!i}" ;;
      --strategy) ((i++)); STRATEGY="${!i}" ;;
      --qr_thresholds) ((i++)); QR_THRESHOLDS="${!i}" ;;
      --shap_thresholds) ((i++)); SHAP_THRESHOLDS="${!i}" ;;
      --shap_start) ((i++)); SHAP_START="${!i}" ;;
      --shap_stop) ((i++)); SHAP_STOP="${!i}" ;;
      --shap_step) ((i++)); SHAP_STEP="${!i}" ;;
      --shap_ncoalitions) ((i++)); SHAP_NCOALITIONS="${!i}" ;;
      --shap_select) ((i++)); SHAP_SELECT_DATA="${!i}" ;;
      --help|-h) usage; exit 0 ;;
      *) : ;;
    esac
  done
}

ensure_data() {
  # Check that the train/test files named in the selected config (configs/$CONFIG.py) exist.
  local missing
  missing=$(cd "$REPO_ROOT" && python - "$CONFIG" <<'PY'
import importlib, os, sys
cfg = importlib.import_module(f"configs.{sys.argv[1]}").config_dict["dataset"]
for key in ("data_file", "test_data_file"):
    path = cfg[key]
    if not os.path.isfile(path):
        print(f"  {key}: {path}")
PY
)
  if [[ -n "$missing" ]]; then
    echo "ERROR: dataset file(s) named in configs/$CONFIG.py not found:" >&2
    echo "$missing" >&2
    exit 1
  fi
}

cmd="${1:-}"; shift || true

case "$cmd" in
  train)
    parse_kv "$@"; ensure_data
    echo "Training config=$CONFIG"
    python "$REPO_ROOT/train_utils/train.py" "$CONFIG"
    ;;

  gen-uncond)
    parse_kv "$@"; ensure_data
    IFS=',' read -r -a seed_arr <<< "$SEEDS"
    out_dir="$UNCOND_OUT_DIR"; [[ "$UNCOND_OUT_DIR" != /* ]] && out_dir="$REPO_ROOT/$UNCOND_OUT_DIR"
    mkdir -p "$out_dir"
    for s in "${seed_arr[@]}"; do
      echo "Unconditional generation: seed=$s config=$CONFIG -> $out_dir"
      python "$REPO_ROOT/inference_utils/unconditional_generation/generate_samples.py" "$s" "$CONFIG" "$out_dir"
    done
    ;;

  pack-uncond)
    parse_kv "$@"
    out_dir="$UNCOND_OUT_DIR"; [[ "$UNCOND_OUT_DIR" != /* ]] && out_dir="$REPO_ROOT/$UNCOND_OUT_DIR"
    mapfile -t files < <(ls -1 "$out_dir"/generated_samples_${CONFIG}_seed*_ep500.npy 2>/dev/null || true)
    if [[ ${#files[@]} -eq 0 ]]; then
      echo "No NPYs found in $out_dir for config=$CONFIG" >&2
      exit 1
    fi
    echo "Packing ${#files[@]} files into HDF5 for config=$CONFIG"
    python "$REPO_ROOT/inference_utils/unconditional_generation/pack_into_hdf5.py" "$CONFIG" "${files[@]}"
    ;;

  gen-cond)
    parse_kv "$@"; ensure_data
    echo "Conditional generation: config=$CONFIG mask_type=$MASK_TYPE mask=$MASK snaps=$SNAPS gen_seed=$GEN_SEED mask_seed=$MASK_SEED random_sensors=${RANDOM_SENSORS:-none}"
    python "$REPO_ROOT/inference_utils/conditional_generation/ddrm-pigdm-mapgd-gen-2D.py"       "$CONFIG" "$MASK_TYPE" "$MASK" "$SNAPS" "$GEN_SEED" "$MASK_SEED" "${RANDOM_SENSORS}"
    ;;

  eval-uncond)
    parse_kv "$@"; ensure_data
    echo "Evaluate unconditional generation: config=$CONFIG num=$NUM results=$RESULTS_DIR"
    python "$REPO_ROOT/evaluate_utils/evaluate-uncond-gen-2D.py" "$CONFIG" "$NUM" "$RESULTS_DIR"
    ;;

  eval-cond)
    parse_kv "$@"; ensure_data
    echo "Evaluate conditional generation: config=$CONFIG mask_type=$MASK_TYPE mask=$MASK snaps=$SNAPS num=$NUM results=$RESULTS_DIR"
    python "$REPO_ROOT/evaluate_utils/evaluate-cond-gen-2D.py"       "$CONFIG" "$MASK_TYPE" "$MASK" "$SNAPS" "$NUM" "$RESULTS_DIR"
    ;;

  shap-values)
    parse_kv "$@"; ensure_data
    echo "Computing SHAP values: config=$CONFIG mask_type=$MASK_TYPE mask=$MASK range=[$SHAP_START,$SHAP_STOP) step=$SHAP_STEP ncoalitions=$SHAP_NCOALITIONS mode=$SHAP_SELECT_DATA"
    python "$REPO_ROOT/osp_utils/shap/obs2D-osp-shap.py"       "$CONFIG" "$MASK_TYPE" "$MASK" "$SHAP_START" "$SHAP_STOP" "$SHAP_STEP" "$SHAP_NCOALITIONS" "$SHAP_SELECT_DATA"
    echo "NOTE: Aggregate SHAP values into threshold masks using the notebooks under osp_utils/shap/related_notebooks/."
    ;;

  shap-summary)
    parse_kv "$@"; ensure_data
    echo "Regenerating SHAP comparison plots from cached datasets"
    python "$REPO_ROOT/osp_utils/evaluate_best/get_shap_results.py"
    ;;

  gen-osp)
    parse_kv "$@"; ensure_data
    driver="$REPO_ROOT/osp_utils/osp-cond-gen-pgdm-mapgd.py"
    if [[ ! -f "$driver" ]]; then
      echo "Driver not found: $driver" >&2; exit 1
    fi
    if [[ "$OSP_METHOD" == "qr" ]]; then
      thresholds="$QR_THRESHOLDS"; strategy="None"
    else
      thresholds="$SHAP_THRESHOLDS"; strategy="$STRATEGY"
    fi
    IFS=',' read -r -a tvals <<< "$thresholds"
    IFS=',' read -r -a seed_arr <<< "$SEEDS"
    echo "Running OSP MAPGD batches: method=$OSP_METHOD thresholds=${thresholds} seeds=${SEEDS} range=($START,$STOP,$STEP) datatype=Test strategy=$strategy"
    for th in "${tvals[@]}"; do
      for s in "${seed_arr[@]}"; do
        echo "Launch: th=$th seed=$s"
        python "$driver" "$CONFIG" "$OSP_METHOD" "$th" "$START" "$STOP" "$STEP" "Test" "$strategy" "$s"
      done
    done
    ;;

  eval-best)
    parse_kv "$@"; ensure_data
    echo "Evaluate OSP best-selection: config=$CONFIG"
    python "$REPO_ROOT/osp_utils/evaluate_best/eval_best.py"
    ;;

  ""|--help|-h|help)
    usage
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
