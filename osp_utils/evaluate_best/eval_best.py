import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure local imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from libs import runner, cond_gen, utils, shap_eval
from configs import *
from libs.cond_gen import plot_mask, plot_sensor_importance

from configs.plot_config import basic_plt_setup,plot_dict
plot_config = runner.dict2namespace(plot_dict)
basic_plt_setup()


CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
            }

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

configname = "OneObs2D_ds1_10M"

print(f"selected config ={configname}")
config_dict = CONFIG_DICT[configname]
config = runner.dict2namespace(config_dict)

osp_dir = Path(config.eval.osp_dir)
shap_dir= osp_dir / "shap"
qr_dir = osp_dir / "qr-pivoting"
threshold_dir = shap_dir / "threshold-masks"


_, u_max, u_min, v_max, v_min, x_axis, y_axis ,t = cond_gen.get_data(config, datatype= "Test" )
num_to_plot=10
ref_value = config.eval.plot_dict.u_reference_value

qr_thresholds=[22.0, 38.0, 71.0] 
gen_seeds_qr = [0, 1, 2, 3, 4, 5]

qr_gen_dir = qr_dir / "generated_samples"

                        
qr_gen_filename_test = "osp-threshold-{thresh}-cond_gen-qr-strategy-None-seed{seed}-M30%-T_sub-1000-T_sub_mapgd-20-gditer_mapgd-50-start-0-stop-25000-step-50-Test.h5"

strategy="mean"
shap_thresholds_v5p5=[0.73, 0.74, 0.75, 0.76, 0.77, 0.775, 0.78, 0.8, 0.82, 0.84, 0.86, 0.9] 
gen_seeds = [0, 1, 2, 3, 4, 5]

version_v5p5 = "v5.5-tdiff20-gd50-snaps-25000-step50"

shap_gen_dir_v5p5 = shap_dir / "generated_samples" / configname / version_v5p5 / strategy

gen_filename_test_v5p5  = "osp-threshold-{thresh}-cond_gen-shap_"+version_v5p5+"-strategy-"+ strategy +"-seed{seed}"+ "-M30%-T_sub-1000-T_sub_mapgd-20-gditer_mapgd-50-start-0-stop-1000-step-1-Test.h5"

results_v5p5 = shap_eval.calculate_error_metrics_with_seeds(thresholds=shap_thresholds_v5p5, gen_seeds=gen_seeds, gen_dir=shap_gen_dir_v5p5, filename=gen_filename_test_v5p5, ref_value=ref_value, pdf_template=None, num_to_plot=num_to_plot, config=config)
results_qr = shap_eval.calculate_error_metrics_with_seeds(thresholds=qr_thresholds, gen_seeds=gen_seeds_qr, gen_dir=qr_gen_dir, filename=qr_gen_filename_test, ref_value=ref_value, pdf_template=None, num_to_plot=num_to_plot, config=config)


random_baseline_results = np.load(str(REPO_ROOT / 'osp_utils' / 'evaluate_best' / 'random_baseline_results_v2_with_best_selection.npz'))
num_pixels_all_sensors=random_baseline_results['num_pixels_all_sensors']
mean_errors_all_sensors =random_baseline_results['mean_errors_all_sensors'] 
std_errors_all_sensors = random_baseline_results['std_errors_all_sensors'] 


datasets_with_best_selection = {
    "SHAP": {
        "num_pixels": results_v5p5["all_num_pixels"],
        "errors": results_v5p5["all_mean_errors"],
        "std_dev": results_v5p5["all_std_errors"],
        "all_mask_tensors": results_v5p5["all_mask_tensors"],
    },
    "Random Baseline": {
        "num_pixels": num_pixels_all_sensors,
        "errors": mean_errors_all_sensors,
        "std_dev": std_errors_all_sensors
    },
    "QR-pivoting": {
        "num_pixels": results_qr["all_num_pixels"],
        "errors": results_qr["all_mean_errors"],
        "std_dev": results_qr["all_std_errors"],
        "all_mask_tensors": results_v5p5["all_mask_tensors"],
    },
}

img_h, img_w = config.model.image_size
total_pixels = img_h * img_w
shap_eval.plot_mixed_binned_and_direct(datasets_with_best_selection, filename="mse_vs_pixels_with_best_selection.pdf", save_dir=str(REPO_ROOT / 'osp_utils' / 'evaluate_best'), total_pixels=total_pixels)


results_v5p5 = shap_eval.calculate_error_metrics_with_seeds_without_best_selection(thresholds=shap_thresholds_v5p5, gen_seeds=gen_seeds, gen_dir=shap_gen_dir_v5p5, filename=gen_filename_test_v5p5, ref_value=1, pdf=None, num_to_plot=num_to_plot, config=config)
results_qr = shap_eval.calculate_error_metrics_with_seeds_without_best_selection(thresholds=qr_thresholds, gen_seeds=gen_seeds, gen_dir=qr_gen_dir, filename=qr_gen_filename_test, ref_value=1, pdf=None, num_to_plot=num_to_plot, config=config)

random_baseline_results = np.load(str(REPO_ROOT / 'osp_utils' / 'evaluate_best' / 'random_baseline_results_v2_without_best_selection.npz'))
num_pixels_all_sensors=random_baseline_results['num_pixels_all_sensors']
mean_errors_all_sensors =random_baseline_results['mean_errors_all_sensors'] 
std_errors_all_sensors = random_baseline_results['std_errors_all_sensors']

datasets_without_best_selection = {
    "SHAP": {
        "num_pixels": results_v5p5["all_num_pixels"],
        "errors": results_v5p5["all_mean_errors"],
        "std_dev": results_v5p5["all_std_errors"],
        "all_mask_tensors": results_v5p5["all_mask_tensors"],
    },
    "Random Baseline": {
        "num_pixels": num_pixels_all_sensors,
        "errors": mean_errors_all_sensors,
        "std_dev": std_errors_all_sensors
    },
    "QR-pivoting": {
        "num_pixels": results_qr["all_num_pixels"],
        "errors": results_qr["all_mean_errors"],
        "std_dev": results_qr["all_std_errors"],
        "all_mask_tensors": results_qr["all_mask_tensors"],
    },
}

shap_eval.plot_mixed_binned_and_direct(datasets_without_best_selection, filename="mse_vs_pixels_without_best_selection.pdf", save_dir=str(REPO_ROOT / 'osp_utils' / 'evaluate_best'), total_pixels=total_pixels)

scale_factor = 100 / (total_pixels)
num_pixels_qr = [np.round(x * scale_factor, 2) for [x] in results_qr['all_num_pixels']]
num_pixels_shap = [np.round(x * scale_factor, 2) for [x] in results_v5p5['all_num_pixels']]

masks_qr = results_qr["all_mask_tensors"]
masks_shap = results_v5p5["all_mask_tensors"]

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
plot_sensor_importance(sensor_importance_array=mask_importance_qr, x_axis=x_axis, y_axis=y_axis, plot_name="QR", num_sensors = ["masked"]+num_pixels_qr, out_show=False, out_pdf=True)
plot_sensor_importance(sensor_importance_array=mask_importance_shap, x_axis=x_axis, y_axis=y_axis, plot_name="SHAP", num_sensors = ["masked"]+num_pixels_shap, out_show=False, out_pdf=True)


# Save
save_dicts = True
if save_dicts:
    import pickle
    with open("datasets_with_best_selection.pkl", "wb") as f:
        pickle.dump(datasets_with_best_selection, f)

    with open("datasets_without_best_selection.pkl", "wb") as f:
        pickle.dump(datasets_without_best_selection, f)