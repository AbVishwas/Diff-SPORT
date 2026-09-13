import numpy as np
import os
import itertools
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import logging
import pandas as pd
from pathlib import Path

from libs import utils, inst_eval, cond_gen
from libs.runner import dict2namespace
from libs import cond_gen
from libs.cond_gen import plot_mask, reduce_sensor_clusters2pixels
from configs.plot_config import plot_dict, basic_plt_setup 

plot_config = dict2namespace(plot_dict)
basic_plt_setup()

################################################################################################
# Helpers moved from eval_best.py (aggregation and plotting)
################################################################################################
def load_file_data(file_path):
    with h5py.File(file_path, 'r') as file:
        gtruth = file["data"][:]
        pred_mapgd = file.get("mapgd")
        if pred_mapgd is not None:
            pred_mapgd = pred_mapgd[:]
        x_axis = file["x"][:]
        y_axis = file["y"][:]
        time = file["t"][:]
        mask_tensor = file["mask_tensor"][:]
    return {
        "gtruth_test": gtruth,
        "pred_mapgd": pred_mapgd,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "time": time,
        "mask_tensor": mask_tensor,
    }

def calculate_error_metrics_with_seeds(thresholds=None, gen_seeds=None, gen_dir=None, filename=None,
                                       ref_value=None, pdf_template=None, num_to_plot=None, config=None):
    metrics = {"inst_errors": None, "predictions": None, "all_mask_tensors": None, "gtruth": None, "x_axis": None, "y_axis": None, "time": None, "all_num_pixels": None, "all_mean_errors": None, "all_std_errors": None, "all_good_fields": None}
    all_best_errors, all_best_preds, all_mask_tensors, all_num_pixels, all_good_fields = [], [], [], [], []
    for thresh in (thresholds or []):
        instance_errors, instance_preds = [], []
        for seed in (gen_seeds or []):
            file_path = Path(gen_dir) / filename.format(thresh=thresh, seed=seed)
            if not file_path.exists():
                logging.warning(f"File not found: {file_path}")
                continue
            data = load_file_data(str(file_path))
            _, _, error_mapgd = utils.calculate_mse(data["gtruth_test"], data["pred_mapgd"], ref_value, std=True)
            error = np.mean(error_mapgd, axis=(1, 2, 3)).reshape(-1, 1)
            instance_errors.append(error)
            instance_preds.append(data["pred_mapgd"])  # (T, C, H, W)
        if len(instance_errors) == 0:
            logging.warning(f"No files found for threshold {thresh}; skipping")
            continue
        instance_errors = np.concatenate(instance_errors, axis=1)  # (T, S)
        instance_preds = np.stack(instance_preds, axis=0)          # (S, T, C, H, W)
        best_indices = np.argmin(instance_errors, axis=1)          # (T,)
        best_errors = instance_errors[np.arange(len(instance_errors)), best_indices].reshape(-1, 1)
        best_preds = instance_preds[best_indices, np.arange(len(best_indices)), ...]
        good_fields = np.count_nonzero(best_errors <= 0.005)
        num_pixels = np.count_nonzero(data["mask_tensor"][0] == 1)
        all_best_errors.append(best_errors); all_best_preds.append(best_preds); all_good_fields.append(good_fields); all_mask_tensors.append(data["mask_tensor"]); all_num_pixels.append(num_pixels)
        if pdf_template:
            try:
                generate_pdf(data["gtruth_test"][:1000], best_preds, data["x_axis"], data["y_axis"], data["time"][:1000], pdf_template.format(thresh=thresh), num_to_plot, config)
            except Exception as e:
                logging.error(f"Failed PDF for threshold {thresh}: {e}")
    if len(all_best_errors) == 0:
        return metrics
    all_best_errors = np.array(all_best_errors, dtype=object); all_best_preds = np.array(all_best_preds, dtype=object); all_mask_tensors = np.array(all_mask_tensors, dtype=object)
    all_num_pixels = np.array(all_num_pixels); all_good_fields = np.array(all_good_fields)
    all_mean_errors = np.array([be.mean() for be in all_best_errors]); all_std_errors = np.array([be.std() for be in all_best_errors])
    metrics.update(inst_errors=all_best_errors, predictions=all_best_preds, all_mask_tensors=all_mask_tensors, gtruth=data["gtruth_test"][:1000], x_axis=data["x_axis"], y_axis=data["y_axis"], time=data["time"][:1000], all_num_pixels=all_num_pixels, all_mean_errors=all_mean_errors, all_std_errors=all_std_errors, all_good_fields=all_good_fields)
    return metrics

def calculate_error_metrics_with_seeds_without_best_selection(thresholds=None, gen_seeds=None, gen_dir=None, filename=None,
                                                              ref_value=None, pdf=None, num_to_plot=None, config=None):
    metrics = {"inst_errors": [], "predictions": [], "all_mask_tensors": [], "gtruth": [], "x_axis": [], "y_axis": [], "time": [], "all_num_pixels": [], "all_mean_errors": [], "all_std_errors": []}
    for thresh in (thresholds or []):
        instance_errors, instance_preds, mask_tensors = [], [], []
        num_pixels_list = []; gtruth_sample = None; x_axis = y_axis = time_sample = None
        for seed in (gen_seeds or []):
            file_path = Path(gen_dir) / filename.format(thresh=thresh, seed=seed)
            if not file_path.exists():
                logging.warning(f"File not found: {file_path}");
                continue
            try:
                data = load_file_data(str(file_path))
                _, _, error_mapgd = utils.calculate_mse(data["gtruth_test"], data["pred_mapgd"], ref_value, std=True)
                error = np.mean(error_mapgd, axis=(1, 2, 3)).reshape(-1, 1)
                instance_errors.append(error); instance_preds.append(data["pred_mapgd"]); mask_tensors.append(data["mask_tensor"])
                if gtruth_sample is None:
                    gtruth_sample = data["gtruth_test"][:1000]; x_axis = data["x_axis"]; y_axis = data["y_axis"]; time_sample = data["time"][:1000]
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        if not instance_errors:
            logging.warning(f"No valid data for threshold {thresh}; skipping");
            continue
        num_pixels_list.append(np.count_nonzero(data["mask_tensor"][0] == 1))
        instance_errors = np.stack(instance_errors, axis=1)  # (T, S, 1)
        instance_preds = np.stack(instance_preds, axis=0); mask_tensors = np.stack(mask_tensors, axis=0)
        mean_errors = np.mean(instance_errors, axis=(0, 1)); std_errors = np.std(instance_errors, axis=(0, 1))
        metrics["inst_errors"].append(instance_errors); metrics["predictions"].append(instance_preds); metrics["all_mask_tensors"].append(mask_tensors)
        metrics["all_num_pixels"].append(np.array(num_pixels_list)); metrics["all_mean_errors"].append(mean_errors); metrics["all_std_errors"].append(std_errors)
        if gtruth_sample is not None:
            metrics["gtruth"].append(gtruth_sample); metrics["x_axis"].append(x_axis); metrics["y_axis"].append(y_axis); metrics["time"].append(time_sample)
        if pdf:
            try:
                generate_pdf(gtruth_sample, instance_preds[:num_to_plot], x_axis, y_axis, time_sample, pdf.format(thresh=thresh), num_to_plot, config)
            except Exception as e:
                logging.error(f"Failed to generate PDF for threshold {thresh}: {e}")
    return metrics

def plot_error_vs_num_sensors(datasets, save_dir=None, out_pdf=False, out_png=False, out_show=False):
    fig, ax1 = plt.subplots(figsize=(8, 6)); ax1.set_xlabel('Number of Pixels'); ax1.set_ylabel('Error')
    ax2 = ax1.twinx(); ax2.set_ylabel('Number of Good Fields')
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    good_field_colors = ['tab:cyan','tab:olive','tab:gray','tab:pink','tab:brown','tab:purple','tab:red','tab:green','tab:orange','tab:blue']
    markers = ['o','s','x','d','^','v','<','>','p','*']
    all_errors, all_good_fields = [], []
    for i,(label,data) in enumerate(datasets.items()):
        num_pixels = np.ravel(data.get('num_pixels',[])); errors = np.ravel(data.get('errors',[])); good_fields = np.ravel(data.get('good_fields',[]))
        color = colors[i%len(colors)]; good_field_color = good_field_colors[i%len(good_field_colors)]; marker = markers[i%len(markers)]
        if num_pixels.size and errors.size:
            ax1.plot(num_pixels, errors, color=color, marker=marker, linestyle='-', label=f'Error-{label}'); all_errors.append(errors)
        if num_pixels.size and good_fields.size:
            ax2.plot(num_pixels, good_fields, color=good_field_color, marker=marker, linestyle='--', label=f'Good Fields-{label}'); all_good_fields.append(good_fields)
    if all_errors:
        ae = np.concatenate(all_errors); ax1.set_ylim(ae.min()*0.9, ae.max()*1.1)
    if all_good_fields:
        ag = np.concatenate(all_good_fields); ax2.set_ylim(ag.min()*0.9, ag.max()*1.1)
    ax1.grid(alpha=0.5); ax1.legend(loc='center right', fontsize=12); ax2.legend(loc='best', fontsize=12)
    plt.title("Error and Good Fields vs. Number of Sensors"); plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True); base = os.path.join(save_dir, 'MSE-compare-v5p5')
    else:
        base = 'MSE-compare-v5p5'
    if out_pdf: plt.savefig(base + '.pdf', dpi=plot_dict['figure']['dpi'])
    if out_png: plt.savefig(base + '.png', dpi=plot_dict['figure']['dpi'])
    if out_show: plt.show()

def plot_mixed_binned_and_direct(datasets, bin_labels=("Random Baseline",), title="Mean Error ± Std Dev vs. Number of Pixels", bins=25, save_dir=None, filename="mse_vs_pixels_without_best_selection.pdf", out_show=False, total_pixels=288*96):
    if bin_labels is None: bin_labels = []
    plt.figure(figsize=(12,10)); plt.xlabel(rf'\% Pixels', fontsize=30); plt.ylabel(r'$\overline\varepsilon$', fontsize=50)
    color_palette=["#1f78b4","#e31a1c","#33a02c","#ffb000","#6a3d9a","#1b9e77","#d95f02","#7570b3","#006400","#e7298a"]; markers=['o','s','x','d','^','v','<','>','p','*']
    for i,(label,data) in enumerate(datasets.items()):
        raw_x = np.ravel(data.get('num_pixels',[])) * 100 / total_pixels; raw_y = np.ravel(data.get('errors',[])); raw_std = np.ravel(data.get('std_dev',[]))
        if raw_x.size==0 or raw_y.size==0 or raw_std.size==0:
            logging.warning(f"Missing data for {label}; skipping"); continue
        color = color_palette[i%len(color_palette)]; marker=markers[i%len(markers)]
        if label in bin_labels:
            df = pd.DataFrame({'x':raw_x,'y':raw_y,'std':raw_std}); df['bin']=pd.cut(df['x'], bins=bins)
            grouped=df.groupby('bin').agg({'x':'mean','y':'mean','std':lambda s: np.sqrt(np.mean(s**2))}).dropna()
            plt.plot(grouped['x'], grouped['y'], color=color, marker=marker, linestyle='-', label=label)
            lower=np.maximum(grouped['y']-grouped['std'],0); upper=grouped['y']+grouped['std']
            plt.fill_between(grouped['x'], lower, upper, color=color, alpha=0.2)
        else:
            plt.plot(raw_x, raw_y, color=color, marker=marker, linestyle='-', label=label)
            lower=np.maximum(raw_y-raw_std,0); upper=raw_y+raw_std
            plt.fill_between(raw_x, lower, upper, color=color, alpha=0.2)
    plt.tick_params(axis='both', labelsize=40); plt.legend(fontsize=30); plt.tight_layout()
    out_path=os.path.join(save_dir or '.', filename); plt.savefig(out_path, dpi=600)
    if out_show: plt.show()



####################################################################
#Evaluation
####################################################################
def normalize_shap_values(shap_values, axis=0):
    """
    Normalizes SHAP values to the range [0, 1] feature-wise.

    Parameters:
    - shap_values: numpy array of SHAP values (features x samples).
    - axis: Axis along which to normalize. Default is 0 (normalize across samples for each feature).

    Returns:
    - shap_normalized: Normalized SHAP values as a numpy array.
    """
    # Validate input
    if not isinstance(shap_values, np.ndarray):
        raise ValueError("shap_values must be a numpy array")
    
    # Compute min and max along the specified axis (feature-wise normalization)
    shap_min = shap_values.min(axis=axis, keepdims=True)
    shap_max = shap_values.max(axis=axis, keepdims=True)
    
    # Normalize values and avoid division by zero
    shap_normalized = (shap_values - shap_min) / (shap_max - shap_min)
    
    return shap_normalized


def compute_mean_shap_values(shap_values=None, feature_wise_norm=True):
    """
    Normalizes SHAP values to [0, 1] feature-wise and computes the mean across samples for each feature.

    Parameters:
    - shap_values: numpy array of shape (num_features, num_samples)
    - feature_wise_norm : bool, optional
        If True, normalize SHAP values across samples for each feature (axis=0).
        If False, normalize SHAP values across features for each sample (axis=1).
    Returns:
    - mean_normalized: Mean of normalized SHAP values across samples for each feature.
    """
    if shap_values is None:
        raise ValueError("shap_values cannot be None. Please provide a numpy array.")

    # Feature-wise normalization
    axis = 0 if feature_wise_norm else 1
    shap_normalized = normalize_shap_values(shap_values, axis=axis)
    
    # Compute mean across samples (axis=1)
    mean_normalized = np.mean(shap_normalized, axis=1)
    
    return mean_normalized


def compute_std_shap_values(shap_values=None, feature_wise_norm=True):
    """
    Normalizes SHAP values to [0, 1] and computes standard deviation across samples.
    
    Parameters:
    - shap_values: numpy array of shape (num_features, num_samples)
    - feature_wise_norm : bool, optional
        If True, normalize SHAP values across samples for each feature (axis=0).
        If False, normalize SHAP values across features for each sample (axis=1).

    Returns:
    - std_normalized: Standard deviation of normalized SHAP values across samples.
    """
    axis = 0 if feature_wise_norm else 1
    shap_normalized = normalize_shap_values(shap_values, axis=axis)

    std_normalized  = np.std(shap_normalized, axis=1)  # Compute std across samples

    return std_normalized



###################################################################
#Mask Creation
###################################################################
def load_shap_values(start=None, stop=None, step=1, bs=1, results_path=None, filename=None):
    """
    Load SHAP values from multiple files in a specified range and concatenate them into a single array.
    
    Parameters:
    -----------
    start : int
        The starting value for the range of files to process.
    stop : int
        The stopping value for the range of files to process (exclusive).
    step : int, optional
        The step size for the range of files to process (default is 1).
    bs : int, optional
        The batch size for file grouping (default is 1).
    results_path : str
        The path to the directory containing the SHAP value files.
    filename : str
        The filename format, which must include placeholders for `start_value` and `stop_value`.
        Example: "shap_{start_value}_{stop_value}.npz"
    
    Returns:
    --------
    all_shaps : np.ndarray
        A concatenated array of SHAP values across all processed files.
    seg_tensor : np.ndarray
        The segmentation tensor from the last successfully loaded file.
    x : np.ndarray
        The x-coordinates from the last successfully loaded file.
    y : np.ndarray
        The y-coordinates from the last successfully loaded file.
    
    Notes:
    ------
    - If a file does not exist or an error occurs during loading, it is skipped, and the error is logged.
    - The function returns data from the last successfully loaded file for `seg_tensor`, `x`, and `y`.
    """
    if results_path is None or filename is None:
        raise ValueError("Both `results_path` and `filename` must be specified.")
    
    counter = 0
    all_shaps = []
    seg_tensor, x, y = None, None, None  # Initialize to ensure they exist
    
    for i in range(start, stop, bs * step):
        stop_value = i + bs * step
        file_path = os.path.join(results_path, filename.format(start_value=i, stop_value=stop_value))
        
        if os.path.exists(file_path):
            try:
                loaded_data = np.load(file_path)
                shaps = loaded_data['shaps']
                
                # Load variables (but only keep necessary ones)
                seg_tensor = loaded_data.get('seg_tensor', None)
                x = loaded_data.get('x', None)
                y = loaded_data.get('y', None)
                
                counter += 1
                all_shaps.append(shaps.reshape(-1, 1))
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {e}")
        else:
            logging.warning(f"File {file_path} does not exist.")
    
    if not all_shaps:
        raise RuntimeError("No SHAP values were loaded. Please check the input parameters and files.")
    
    all_shaps = np.abs(np.concatenate(all_shaps, axis=1))
    return all_shaps, seg_tensor, x, y

def get_sensors_for_each_batch_legacy(
    all_shaps=None,
    threshold=0.6,
    seg_tensor=None,
    mask=30,
    x_axis=None,
    y_axis=None,
    feature_wise_norm=True,
    plot_shap_values=False,
    plot_masks=False
):
    """
    Computes sensor placement arrays and SHAP matrices for each batch based on normalized SHAP values.

    Parameters:
    ----------
    all_shaps : np.ndarray
        A 2D array of SHAP values with shape (features, num_batches).
    threshold : float, optional
        Threshold value to determine sensor placement (default is 0.6).
    seg_tensor : np.ndarray
        A segmentation tensor used to map SHAP values to a 2D space.
    mask : int, optional
        A mask value used during plotting (default is 30).
    x_axis : np.ndarray, optional
        The x-coordinates for plotting (default is None).
    y_axis : np.ndarray, optional
        The y-coordinates for plotting (default is None).
    feature_wise_norm : bool, optional
        If True, normalizes SHAP values across samples for each feature (default is True).
    plot_shap_values : bool, optional
        If True, plots the SHAP value matrices (default is False).
    plot_masks : bool, optional
        If True, plots the sensor placement arrays (default is False).

    Returns:
    -------
    all_shap_mats : np.ndarray
        A 3D array containing SHAP value matrices for all batches (num_batches, x_dim, y_dim).
    sensor_placement_arrays : np.ndarray
        A 3D array containing binary sensor placement arrays for all batches (num_batches, x_dim, y_dim).
    """
    if all_shaps is None or seg_tensor is None:
        raise ValueError("Both `all_shaps` and `seg_tensor` must be provided.")
    
    if x_axis is None or y_axis is None:
        raise ValueError("Both `x_axis` and `y_axis` must be provided for plotting.")

    # Normalize SHAP values
    if feature_wise_norm:
        all_shaps = normalize_shap_values(all_shaps, axis=0)

    all_shap_mats = []
    sensor_placement_arrays = []

    # Get unique segment values and their counts
    unique_values, counts = np.unique(seg_tensor, return_counts=True)

    for j in range(all_shaps.shape[1]):  # Loop through all samples
        shap_mat = np.zeros_like(seg_tensor)

        for i in range(all_shaps.shape[0]):  # Loop through all features
            shap_mat[seg_tensor == unique_values[i + 1]] = all_shaps[i, j]

        all_shap_mats.append(shap_mat)

        # Generate sensor placement array
        sensor_placement_array = np.where(shap_mat < threshold, 0, 1)
        sensor_placement_arrays.append(sensor_placement_array)

        # Plot SHAP values if enabled
        if plot_shap_values:
            plot_mask(
                mask_tensor=shap_mat,
                mask=mask,
                x_axis=x_axis,
                y_axis=y_axis,
                mask_type=f"shap-{j}",
                cmap="viridis",
                out_pdf=True,
            )

        # Plot sensor placements if enabled
        if plot_masks:
            plot_mask(
                mask_tensor=sensor_placement_array,
                mask=mask,
                x_axis=x_axis,
                y_axis=y_axis,
                mask_type=f"sensor-{j}",
                cmap="viridis",
                out_pdf=True,
            )

    # Convert lists to arrays
    all_shap_mats = np.array(all_shap_mats)
    sensor_placement_arrays = np.array(sensor_placement_arrays)

    return all_shap_mats, sensor_placement_arrays


def generate_all_shap_mats(
    all_shaps=None,
    seg_tensor=None,
    mask=None,
    feature_wise_norm=True,
    plot_shap_mats=False,
    x_axis=None,
    y_axis=None
):
    """
    Generates SHAP matrices for each batch based on normalized SHAP values.

    Parameters:
    ----------
    all_shaps : np.ndarray
        A 2D array of SHAP values with shape (features, num_batches).
    seg_tensor : np.ndarray
        A segmentation tensor used to map SHAP values to a 2D space.
    feature_wise_norm : bool, optional
        If True, normalizes SHAP values across samples for each feature (default is True).
    plot_shap_mats : bool, optional
        If True, plots the SHAP matrices (default is False).
    x_axis : np.ndarray, optional
        The x-coordinates for plotting (default is None).
    y_axis : np.ndarray, optional
        The y-coordinates for plotting (default is None).

    Returns:
    -------
    all_shap_mats : np.ndarray
        A 3D array containing SHAP value matrices for all batches (num_batches, x_dim, y_dim).
    """
    if all_shaps is None or seg_tensor is None:
        raise ValueError("Both `all_shaps` and `seg_tensor` must be provided.")

    if plot_shap_mats and (x_axis is None or y_axis is None):
        raise ValueError("Both `x_axis` and `y_axis` must be provided for plotting.")

    # Normalize SHAP values
    if feature_wise_norm:
        all_shaps = normalize_shap_values(all_shaps, axis=0)

    all_shap_mats = []

    # Get unique segment values
    unique_values, counts = np.unique(seg_tensor, return_counts=True)

    for j in range(all_shaps.shape[1]):  # Loop through all samples
        shap_mat = np.zeros_like(seg_tensor)

        for i in range(all_shaps.shape[0]):  # Loop through all features
            shap_mat[seg_tensor == unique_values[i + 1]] = all_shaps[i, j]

        all_shap_mats.append(shap_mat)

        # Plot SHAP matrices if enabled
        if plot_shap_mats:
            plot_mask(
                mask_tensor=shap_mat,
                mask=mask,
                x_axis=x_axis,
                y_axis=y_axis,
                mask_type=f"shap-matrix-{j}",
                cmap="viridis",
                out_pdf=True,
            )

    # Convert list to array
    all_shap_mats = np.array(all_shap_mats)

    return all_shap_mats

def generate_sensors_per_batch(
    all_shap_mats=None,
    threshold=0.6,
    plot_masks=False,
    mask=30,
    x_axis=None,
    y_axis=None
):
    """
    Generates sensor placement arrays based on SHAP matrices and a threshold.

    Parameters:
    ----------
    all_shap_mats : np.ndarray
        A 3D array containing SHAP value matrices for all batches (num_batches, x_dim, y_dim).
    threshold : float, optional
        Threshold value to determine sensor placement (default is 0.6).
    plot_masks : bool, optional
        If True, plots the sensor placement arrays (default is False).
    mask : int, optional
        A mask value used during plotting (default is 30).
    x_axis : np.ndarray, optional
        The x-coordinates for plotting (default is None).
    y_axis : np.ndarray, optional
        The y-coordinates for plotting (default is None).

    Returns:
    -------
    sensor_placement_arrays : np.ndarray
        A 3D array containing binary sensor placement arrays for all batches (num_batches, x_dim, y_dim).
    """
    if all_shap_mats is None:
        raise ValueError("`all_shap_mats` must be provided.")

    if plot_masks and (x_axis is None or y_axis is None):
        raise ValueError("Both `x_axis` and `y_axis` must be provided for plotting.")

    sensor_placement_arrays = []

    for j, shap_mat in enumerate(all_shap_mats):
        # Generate sensor placement array
        sensor_placement_array = np.where(shap_mat < threshold, 0, 1)
        sensor_placement_arrays.append(sensor_placement_array)

        # Plot sensor placements if enabled
        if plot_masks:
            plot_mask(
                mask_tensor=sensor_placement_array,
                mask=mask,
                x_axis=x_axis,
                y_axis=y_axis,
                mask_type=f"sensor-{j}",
                cmap="viridis",
                out_pdf=True,
            )

    # Convert list to array
    sensor_placement_arrays = np.array(sensor_placement_arrays)

    return sensor_placement_arrays

def mean_plus_std_shap_vals(all_shap_mats=None):
    mean = np.mean(all_shap_mats, axis=0)
    std = np.std(all_shap_mats, axis=0)

    normalized_shap_mat = mean + std

    return normalized_shap_mat/np.max(normalized_shap_mat)

def mean_shap_vals(all_shap_mats=None):
    mean = np.mean(all_shap_mats, axis=0)
    return mean/np.max(mean)

def create_shap_masks(all_shap_mats=None, strategy=None, threshold=0.5, mask_value=30, 
                 x_axis=None, y_axis=None, plot_final_mask=False):
    """
    Generate sensor placement masks based on SHAP values.

    Parameters:
    - all_shap_mats (numpy.ndarray): SHAP values matrix.
    - strategy (str): Mask generation strategy ('MultiMode' or 'mean_plus_std').
    - threshold (float): Threshold for masking.
    - mask_value (int): Mask size for 'MultiMode'.
    - x_axis (numpy.ndarray): X-axis coordinates for plotting.
    - y_axis (numpy.ndarray): Y-axis coordinates for plotting.
    - plot_final_mask (bool): Whether to plot the final mask.

    Returns:
    - final_sensor_array (numpy.ndarray): The final sensor mask.
    - reduced_shap_mask (numpy.ndarray): The reduced mask after clustering.
    """
    if strategy not in ["MultiMode", "mean_plus_std", "mean"]:
        raise ValueError(f"Invalid strategy '{strategy}'. Choose 'MultiMode' or 'mean_plus_std' or 'mean'.")
    if all_shap_mats is None:
        raise ValueError("Parameter 'all_shap_mats' must not be None.")

    if strategy == "MultiMode":
        assert all_shap_mats.max() == 1, "For multimodal mask generation, all_shap_mats must be normalized."
        sensor_arrays = generate_sensors_per_batch(
            all_shap_mats=all_shap_mats,
            threshold=threshold,
            mask=mask_value,
            x_axis=x_axis,
            y_axis=y_axis,
            plot_masks=True
        )
        final_sensor_array = np.sum(sensor_arrays, axis=0)
        final_sensor_array[final_sensor_array >= 1] = 1

    elif strategy == "mean_plus_std":
        avg_norm_shap_vals = mean_plus_std_shap_vals(all_shap_mats=all_shap_mats)
        final_sensor_array = np.where(avg_norm_shap_vals < threshold, 0, 1)

        if plot_final_mask:
            plot_mask(
                mask_tensor=avg_norm_shap_vals,
                mask=mask_value,
                x_axis=x_axis,
                y_axis=y_axis,
                cmap="viridis",
                mask_type=f"shapvals-{strategy}",
                out_pdf=True
            )            

    elif strategy == "mean":
        avg_norm_shap_vals = mean_shap_vals(all_shap_mats=all_shap_mats)
        final_sensor_array = np.where(avg_norm_shap_vals < threshold, 0, 1)

        if plot_final_mask:
            plot_mask(
                mask_tensor=avg_norm_shap_vals,
                mask=mask_value,
                x_axis=x_axis,
                y_axis=y_axis,
                cmap="viridis",
                mask_type=f"shapvals-{strategy}",
                out_pdf=True
            )  

    reduced_shap_mask = reduce_sensor_clusters2pixels(mask_tensor=final_sensor_array)

    # Debugging Information
    num_initial_pixels = np.count_nonzero(final_sensor_array[0] == 1)
    print(f"Number of pixels in the initial mask: {num_initial_pixels}")
    num_reduced_pixels = np.count_nonzero(reduced_shap_mask[0] == 1)
    print(f"Number of reduced pixels in the mask: {num_reduced_pixels}")

    if plot_final_mask:
        plot_mask(
            mask_tensor=final_sensor_array,
            mask=mask_value,
            x_axis=x_axis,
            y_axis=y_axis,
            mask_type=f"sensor-{strategy}-thresh-{threshold}",
            out_pdf=True
        )

    return final_sensor_array, reduced_shap_mask

def aggregate_normalize_all_batch_shap_mats_legacy(all_batch_data =None, threshold=0.6,
                                    x_axis=None, y_axis=None, mask=30, 
                                    plot_masks=False):

    all_batch_data_sum_norm  = np.mean(np.expand_dims(all_batch_data, axis = 0), axis=0)
 
    if plot_masks:
        plot_mask(mask_tensor=all_batch_data_sum_norm[0], mask=mask,
                  x_axis=x_axis, y_axis=y_axis, mask_type=f"shapvals-avg-thresh-{threshold}", cmap="viridis",out_pdf=True)

    return all_batch_data_sum_norm


def aggregate_normalize_all_batch_shap_masks_legacy(sensor_arrays =None, threshold=0.6,
                                    x_axis=None, y_axis=None, mask=30, 
                                    reduced_mask=True, plot_masks=False):
    
    sensor_arrays_sum  = np.sum(sensor_arrays, axis=0)
    sensor_arrays_sum[sensor_arrays_sum >= 1] = 1

    num_pixels = np.count_nonzero( sensor_arrays_sum[0] == 1 )
    print(f"number of pixels in the mask: {num_pixels}")
    
    if plot_masks:
        plot_mask(mask_tensor=sensor_arrays_sum, mask=mask,
                  x_axis=x_axis, y_axis=y_axis, mask_type=f"sensor-avg-thresh-{threshold}", out_pdf=True)

    if reduced_mask:
        reduced_shap_mask = reduce_sensor_clusters2pixels(mask_tensor=sensor_arrays_sum)
        plot_mask(mask_tensor=reduced_shap_mask, mask=mask, x_axis=x_axis, y_axis=y_axis, mask_type=f"red-sensor-avg-thresh-{threshold}", out_pdf=True)

        
        num_pixels_red = np.count_nonzero( reduced_shap_mask[0] == 1 )
        
        print(f"number of reduced pixels in the mask: {num_pixels_red}")

        return sensor_arrays_sum, reduced_shap_mask
    return sensor_arrays_sum


def get_train_mean(config=None):
    data, u_max, u_min, v_max, v_min, x, y ,t = cond_gen.get_data(config, datatype= "Train")
    #data = cond_gen.normalize(data, u_max, u_min, v_max, v_min)

    train_mean = torch.from_numpy(np.mean(data, axis=0, keepdims=True)).to("cuda")
    print(f"Shape of train mean:{train_mean.shape}")
    return train_mean



##################################################################
#Error Calculation
##################################################################
def load_threshold_masks(thresholds=None, dir=None, filename=None, mask_type="shap"):
    """
    Load threshold masks with support for different file formats and naming conventions.

    Parameters:
        thresholds (list): List of threshold values or sensor counts.
        dir (str): Directory containing the mask files.
        filename (str): Filename pattern with placeholders for threshold/sensor count.
        mask_type (str): Type of mask to load ("shap" or "qr").

    Returns:
        mask_array (np.ndarray): Array of mask tensors.
        num_pixels (list): List of nonzero pixels or sensors per threshold.
        importance (np.ndarray): Aggregated importance map (SHAP_importance or QR_importance).
    """
    mask_array = []
    num_pixels = []

    for thresh in thresholds:
        filepath = dir + filename.format(thresh=thresh)

        # Load mask file
        mask_file = np.load(filepath)

        if mask_type == "shap":
            mask_tensor = mask_file["shap_mask"]
        elif mask_type == "qr":
            mask_tensor = mask_file["qr_mask"]
        else:
            raise ValueError("Unsupported mask_type. Use 'shap' or 'qr'.")

        mask = mask_file["mask"]
        num_pixel = np.count_nonzero(mask_tensor[0] == 1)
        print(f"Number of pixels for {mask_type} thresh {thresh} = {num_pixel}")

        mask_array.append(mask_tensor)
        num_pixels.append(num_pixel)

    mask_array = np.concatenate(np.expand_dims(mask_array, axis=0), axis=0)
    importance = np.sum(mask_array, axis=0)
    num_pixels = list(num_pixels) 

    return mask_array, num_pixels, importance

def load_file_data(file_path):
    """Load data from the given file path."""
    with h5py.File(file_path, 'r') as pred_file:
        return {
            "gtruth_test": pred_file["data"][:],
            #"pred_ddrm": pred_file["ddrm"][:],
            #"pred_pgdm": pred_file["pgdm"][:],
            "pred_mapgd": pred_file["mapgd"][:],
            "mask_tensor": pred_file["mask_tensor"][:],
            "x_axis": pred_file["x"][:],
            "y_axis": pred_file["y"][:],
            "time": pred_file["t"][:],
        }

def aggregate_metrics(metrics):
    """Concatenate and compute final metrics."""
    return {key: np.concatenate(values, axis=0) for key, values in metrics.items()}

def generate_pdf(gtruth_test, pred, x_axis, y_axis, time, pdf_path, num_to_plot, config):
    """Generate and save PDF plots."""
    with PdfPages(pdf_path) as pdf:
        inst_evaluator = inst_eval.InstantaneousEvaluation(
            gtruth=gtruth_test,
            pred1=pred,
            pred2=None,
            x_axis=x_axis,
            y_axis=y_axis,
            time=time[: gtruth_test.shape[0]],
            input_data_type="2D",
            config=config,
        )
        random_indices = utils.select_random(
            gtruth_test, num_elements=num_to_plot, seed=0, only_indices=True
        )
        inst_evaluator.main(random_indices=random_indices, mask=30, ds_ratio=1, pdf=pdf)


def calculate_error_metrics(thresholds=None, gen_dir=None, filename=None, ref_value=None, pdf=None, num_to_plot=None, config=None):
    """Main function to calculate error metrics."""
    metrics = {
        "error_mapgd_mse_mean": [],
        "error_mapgd_mse_std": [],  # Added for standard deviation
        "error_mapgd_mse_total": [],
        "num_pixels": [],
        "mask_tensors": [],
    }

    for i, thresh in enumerate(thresholds):
        
        file_path = os.path.join(gen_dir, filename.format(thresh=thresh))

        if os.path.exists(file_path):
            data = load_file_data(file_path)
            num_pixel = np.count_nonzero(data["mask_tensor"][0] == 1)
            metrics["num_pixels"].append(np.expand_dims(num_pixel, axis=0))
            
            # Calculate mean and std using utils.calculate_mse
            error_mapgd_mse, error_mapgd_mse_std, error_mapgd = utils.calculate_mse(
                data["gtruth_test"], data["pred_mapgd"], ref_value, std=True
            )

            print(rf"Threshold: {thresh}, Pixels: {num_pixel}, Error:{np.mean(error_mapgd_mse)}")

            metrics["error_mapgd_mse_mean"].append(np.expand_dims(error_mapgd_mse, axis=0))
            metrics["error_mapgd_mse_std"].append(np.expand_dims(error_mapgd_mse_std, axis=0))  # Store std
            metrics["error_mapgd_mse_total"].append(np.expand_dims(error_mapgd, axis=0))
            metrics["mask_tensors"].append(np.expand_dims(data["mask_tensor"], axis=0))

            if pdf:
                generate_pdf(
                    data["gtruth_test"],
                    data["pred_mapgd"],
                    data["x_axis"],
                    data["y_axis"],
                    data["time"],
                    pdf.format(thresh=thresh),
                    num_to_plot,
                    config,
                )
        else:
            print(f"File not found: {filename.format(thresh=thresh)}")

    aggregated_metrics = aggregate_metrics(metrics)
    error_mapgd_mse_mean_value = np.mean(aggregated_metrics["error_mapgd_mse_mean"], axis=(2, 3))
    error_mapgd_mse_std_value = np.mean(aggregated_metrics["error_mapgd_mse_std"], axis=(2, 3))  # Aggregate std

    return {
        **aggregated_metrics,
        "error_mapgd_mse_mean_value": error_mapgd_mse_mean_value,
        "error_mapgd_mse_std_value": error_mapgd_mse_std_value,  # Return aggregated std
    }

##################################################################
#Plotting 
##################################################################


def plot_shap_distributions(shap_data_dict, bins=100, alpha=0.5, density=True, ylim=(0, 10), feature_wise_norm=True, figname=None, out_show=False, out_pdf=False, out_png=False):
    """
    Plots distributions of normalized SHAP values from multiple datasets.

    Parameters:
    ----------
    shap_data_dict : dict
        A dictionary where keys are labels for the legend, and values are arrays of SHAP values.
    bins : int, optional
        Number of bins for the histogram (default is 100).
    alpha : float, optional
        Transparency level for the histograms (default is 0.5).
    density : bool, optional
        If True, normalize the histogram (default is True).
    ylim : tuple, optional
        Y-axis limits for the plot (default is (0, 10)).
    figname : str, optional
        Name identifier for saved files (default is None).
    out_show : bool, optional
        If True, displays the plot (default is False).
    out_pdf : bool, optional
        If True, saves the plot as a PDF file (default is False).
    out_png : bool, optional
        If True, saves the plot as a PNG file (default is False).

    Returns:
    -------
    None
    """
    if not isinstance(shap_data_dict, dict):
        raise ValueError("Input shap_data_dict must be a dictionary.")

    plt.figure(figsize=(8, 6))
    axis = 0 if feature_wise_norm else 1

    for label, data in shap_data_dict.items():
        if not isinstance(data, np.ndarray):
            raise ValueError(f"All values in shap_data_dict must be numpy arrays. Found {type(data)} for label '{label}'.")

        data_norm = normalize_shap_values(data, axis=axis)

        # Flatten the data and plot histogram
        plt.hist(data_norm.flatten(), bins=bins, alpha=alpha, label=label, density=density)


    plt.title("Normalized SHAP Value Distributions")
    plt.xlabel("Normalized SHAP Value")
    plt.ylabel("Density")
    plt.ylim(ylim)
    plt.legend()

    if out_show:
        plt.show()

    if out_pdf and figname:
        plt.savefig(f"SHAP-normalized-distributions-{figname}.pdf", dpi=300)
    
    if out_png and figname:
        plt.savefig(f"SHAP-normalized-distributions-{figname}.png", dpi=300)
    
    plt.close()

def plot_shap_std_deviations(shap_data_dict, feature_wise_norm=True, figname=None, out_show=False, out_pdf=False, out_png=False):
    """
    Plots the standard deviation of normalized SHAP values for multiple datasets.

    Parameters:
    ----------
    shap_data_dict : dict
        A dictionary where keys are labels for the legend, and values are arrays of SHAP values.
    feature_wise_norm : bool, optional
        If True, normalize SHAP values across samples for each feature (default is True).
    figname : str, optional
        Name identifier for saved files (default is None).
    out_show : bool, optional
        If True, displays the plot (default is False).
    out_pdf : bool, optional
        If True, saves the plot as a PDF file (default is False).
    out_png : bool, optional
        If True, saves the plot as a PNG file (default is False).

    Returns:
    -------
    None
    """
    if not isinstance(shap_data_dict, dict):
        raise ValueError("Input shap_data_dict must be a dictionary.")

    plt.figure(figsize=(10, 6))

    for label, data in shap_data_dict.items():
        if not isinstance(data, np.ndarray):
            raise ValueError(f"All values in shap_data_dict must be numpy arrays. Found {type(data)} for label '{label}'.")

        std_dev = compute_std_shap_values(data, feature_wise_norm=feature_wise_norm)
        plt.scatter(range(len(std_dev)), std_dev , label=label, s=5)

    plt.title("Normalized SHAP Value Variability Across Features")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Standard Deviation of SHAP Values")
    plt.legend()

    if out_show:
        plt.show()

    if out_pdf and figname:
        plt.savefig(f"SHAP-std-dev-{figname}.pdf", dpi=300)
    
    if out_png and figname:
        plt.savefig(f"SHAP-std-dev-{figname}.png", dpi=300)
    
    plt.close()


def plot_shap_means(shap_data_dict, feature_wise_norm=True, figname=None, out_show=False, out_pdf=False, out_png=False):
    """
    Plots the mean of normalized SHAP values for multiple datasets.

    Parameters:
    ----------
    shap_data_dict : dict
        A dictionary where keys are labels for the legend, and values are arrays of SHAP values.
    feature_wise_norm : bool, optional
        If True, normalize SHAP values across samples for each feature (default is True).
    figname : str, optional
        Name identifier for saved files (default is None).
    out_show : bool, optional
        If True, displays the plot (default is False).
    out_pdf : bool, optional
        If True, saves the plot as a PDF file (default is False).
    out_png : bool, optional
        If True, saves the plot as a PNG file (default is False).

    Returns:
    -------
    None
    """
    if not isinstance(shap_data_dict, dict):
        raise ValueError("Input shap_data_dict must be a dictionary.")

    plt.figure(figsize=(10, 6))

    for label, data in shap_data_dict.items():
        if not isinstance(data, np.ndarray):
            raise ValueError(f"All values in shap_data_dict must be numpy arrays. Found {type(data)} for label '{label}'.")

        mean_values = compute_mean_shap_values(data, feature_wise_norm=feature_wise_norm)
        plt.scatter(range(len(mean_values)), mean_values, label=label, s=5)

    plt.title("Mean Normalized SHAP Values Across Features")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Normalized SHAP Values")
    plt.legend()

    if out_show:
        plt.show()

    if out_pdf and figname:
        plt.savefig(f"SHAP-mean-{figname}.pdf", dpi=300)
    
    if out_png and figname:
        plt.savefig(f"SHAP-mean-{figname}.png", dpi=300)
    plt.close()


def print_top_features(shap_data_dict, k=10, feature_wise_norm=True):
    """
    Prints the top-k features with the highest mean normalized SHAP values for each dataset.

    Parameters:
    ----------
    shap_data_dict : dict
        A dictionary where keys are labels for the datasets, and values are arrays of SHAP values.
    k : int, optional
        Number of top features to display (default is 10).
    feature_wise_norm : bool, optional
        If True, normalize SHAP values across samples for each feature (default is True).

    Returns:
    -------
    None
    """
    if not isinstance(shap_data_dict, dict):
        raise ValueError("Input shap_data_dict must be a dictionary.")

    for label, data in shap_data_dict.items():
        if not isinstance(data, np.ndarray):
            raise ValueError(f"All values in shap_data_dict must be numpy arrays. Found {type(data)} for label '{label}'.")

        mean_values = compute_mean_shap_values(data, feature_wise_norm=feature_wise_norm)

        ranked_features= np.argsort(-mean_values)
        top_k = ranked_features[:k]
        print(f"Top-{k} features for ({label}): {top_k}")


def plot_random_indices(data=None, num_indices=5, seed=0):
    """
    Plots `num_indices` random elements from `data` across axis=1.

    Parameters:
        data (ndarray): The input 2D array.
        num_indices (int): Number of random indices to plot.
    """
    plt.figure(figsize=(25, 12.5))
    np.random.seed(seed)
    for _ in range(num_indices):
        random_index = np.random.randint(0, data.shape[0])
        selected_element = data[random_index, :]
        plt.plot(np.linspace(23000, 25000, selected_element.shape[0]), selected_element, marker='o', linestyle='-', label=f'Index {random_index}')

    plt.title(f"{num_indices} Randomly Selected Elements")
    plt.xlabel("Number of snapshots")
    plt.ylabel("Normalized cumulative SHAP Value")
    plt.grid(True)
    #plt.legend(loc="right")
    #plt.ylim(-0.01, 0.3)

    if out_show:
        plt.show()

    if out_pdf and figname:
        plt.savefig(f"SHAP-value-convergence-{figname}.pdf", dpi=300)
    
    if out_png and figname:
        plt.savefig(f"SHAP-value-convergence-{figname}.png", dpi=300)
    plt.close()


def plot_mse_osp_comparison(output_dict=None,
    convert_pixel2sensors=False,
    out_show=False,
    out_pdf=False,
    out_png=False):
    """
    Plots and compares MSE for different sensor placement methods.

    Parameters:
    ----------
    output_dict : dict
        A dictionary where keys represent method names (e.g., 'v2.5-trn', 'v2.5-tst'),
        and values are dictionaries containing MSE values and number of pixels.
    convert_pixel2sensors : bool, optional
        If True, converts pixels to sensors by dividing by 6.
    out_show : bool, optional
        If True, displays the plot.
    out_pdf : bool, optional
        If True, saves the plot as a PDF.
    out_png : bool, optional
        If True, saves the plot as a PNG.

    Returns:
    -------
    None
    """
    def plot_method(data_dict, color, label, marker_circle, marker_square, linestyle):
        """Helper function to plot a single method."""
        num_sensors = data_dict["num_pixels"] // 6 if convert_pixel2sensors else data_dict["num_pixels"]
        
        # Plot mean and std deviation band for u'
        ax1.plot(num_sensors, data_dict["error_mapgd_mse_mean_value"][:, 0], color=color, linestyle=linestyle)
        ax1.fill_between(
            num_sensors,
            data_dict["error_mapgd_mse_mean_value"][:, 0] - data_dict["error_mapgd_mse_std_value"][:, 0],
            data_dict["error_mapgd_mse_mean_value"][:, 0] + data_dict["error_mapgd_mse_std_value"][:, 0],
            color=color,
            alpha=0.2
        )
        ax1.scatter(num_sensors, data_dict["error_mapgd_mse_mean_value"][:, 0], color=color, marker=marker_circle, s=s, label=f"{label} $u'$")
        
        # Plot mean and std deviation band for v'
        ax1.plot(num_sensors, data_dict["error_mapgd_mse_mean_value"][:, 1], color=color, linestyle=linestyle)
        ax1.fill_between(
            num_sensors,
            data_dict["error_mapgd_mse_mean_value"][:, 1] - data_dict["error_mapgd_mse_std_value"][:, 1],
            data_dict["error_mapgd_mse_mean_value"][:, 1] + data_dict["error_mapgd_mse_std_value"][:, 1],
            color=color,
            alpha=0.2
        )
        ax1.scatter(num_sensors, data_dict["error_mapgd_mse_mean_value"][:, 1], color=color, marker=marker_square, s=s, label=f"{label} $v'$")

    # Setup
    fig, ax1 = plt.subplots(figsize=(10, 8))
    s = 50
    xlabel = "Sensors (Pixels/6)" if convert_pixel2sensors else "Pixels"

    # Generate distinct colors for each method
    color_palette = plt.cm.tab10.colors
    color_map = {key: color_palette[i % len(color_palette)] for i, key in enumerate(output_dict.keys())}
    legend_lines = []

    # Plot data for each method
    if output_dict:
        for method, data in output_dict.items():
            if data:
                plot_method(data, color_map[method], method, "o", "s", "--")
                line = mlines.Line2D([], [], color=color_map[method], linestyle="--", label=method)
                legend_lines.append(line)

    # Labels and grid
    ax1.set_xlabel(xlabel, fontsize=15)
    ax1.set_ylabel(r"$\varepsilon$", fontsize=15)
    ax1.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Custom Legend
    circle = mlines.Line2D([], [], markerfacecolor="none", markeredgecolor="k", marker="o", linestyle="None", label=r"$u'$")
    square = mlines.Line2D([], [], markerfacecolor="none", markeredgecolor="k", marker="s", linestyle="None", label=r"$v'$")
    ax1.legend(handles=legend_lines + [circle, square], loc="upper right", fontsize=14)

    # Save or show plot
    if out_show:
        plt.show()
    if out_pdf:
        plt.savefig(f"mse-vs-{xlabel.replace(' ', '_')}.pdf", dpi=300)
    if out_png:
        plt.savefig(f"mse-vs-{xlabel.replace(' ', '_')}.png", dpi=300)
    plt.close()
