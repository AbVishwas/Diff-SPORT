#!/usr/bin/env python3
"""
plot_saved_dicts.py

Load the saved pickles:
  - datasets_with_best_selection.pkl
  - datasets_without_best_selection.pkl

and plot them directly using shap_eval.plot_mixed_binned_and_direct.
"""

import pickle
import sys
from pathlib import Path
import numpy as np

# Detect repo root (same as your main script)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from libs import runner, shap_eval, cond_gen
from configs import OneObs2D_ds1_10M  # adjust if you use other configs


# Choose config here
configname = "OneObs2D_ds1_10M"
CONFIG_DICT = {
    "OneObs2D_ds1_10M": OneObs2D_ds1_10M.config_dict,
}
config_dict = CONFIG_DICT[configname]
config = runner.dict2namespace(config_dict)

# total pixels from image size
img_h, img_w = config.model.image_size
total_pixels = img_h * img_w

# paths
save_dir = REPO_ROOT / "osp_utils" / "evaluate_best"
pkl_with = save_dir / "datasets_with_best_selection.pkl"
pkl_without = save_dir / "datasets_without_best_selection.pkl"

# load pickles
datasets_with = pickle.load(open(pkl_with, "rb")) if pkl_with.exists() else None
datasets_without = pickle.load(open(pkl_without, "rb")) if pkl_without.exists() else None

# plot
if datasets_with:
    shap_eval.plot_mixed_binned_and_direct(
        datasets_with,
        filename="mse_vs_pixels_with_best_selection.pdf",
        save_dir=str(save_dir),
        total_pixels=total_pixels,
    )
    print("Saved mse_vs_pixels_with_best_selection.pdf")

if datasets_without:
    shap_eval.plot_mixed_binned_and_direct(
        datasets_without,
        filename="mse_vs_pixels_without_best_selection.pdf",
        save_dir=str(save_dir),
        total_pixels=total_pixels,
    )
    print("Saved mse_vs_pixels_without_best_selection.pdf")

    scale_factor = 100 / (total_pixels)
    num_pixels_qr = [np.round(x * scale_factor, 2) for [x] in datasets_without["QR-pivoting"]['num_pixels']]
    num_pixels_shap = [np.round(x * scale_factor, 2) for [x] in datasets_without["SHAP"]['num_pixels']]

    masks_qr = datasets_without["QR-pivoting"]["all_mask_tensors"]
    masks_shap = datasets_without["SHAP"]["all_mask_tensors"]

    final_mask_tensors_qr = []
    for mask in masks_qr:
        gen_seed_mask = np.mean(mask, axis=0)
        final_mask_tensors_qr.append(gen_seed_mask)

    final_mask_tensors_shap = []
    for mask in masks_shap:
        gen_seed_mask = np.mean(mask, axis=0)
        final_mask_tensors_shap.append(gen_seed_mask)

    mask_importance_qr = np.sum(final_mask_tensors_qr, axis =0)
    mask_importance_shap = np.sum(final_mask_tensors_shap, axis =0)

    #plot_error_vs_num_sensors(datasets, out_pdf=True)
    _, _, _, _, _, x_axis, y_axis ,t = cond_gen.get_data(config, datatype= "Test" )
    cond_gen.plot_sensor_importance(sensor_importance_array=mask_importance_qr, x_axis=x_axis, y_axis=y_axis, plot_name="QR", num_sensors = ["masked"]+num_pixels_qr, out_show=False, out_pdf=True)
    cond_gen.plot_sensor_importance(sensor_importance_array=mask_importance_shap, x_axis=x_axis, y_axis=y_axis, plot_name="SHAP", num_sensors = ["masked"]+num_pixels_shap, out_show=False, out_pdf=True)
