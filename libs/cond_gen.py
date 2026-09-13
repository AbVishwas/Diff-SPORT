import numpy as np
import torch
import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
sys.path.append('../../')

from libs.runner import dict2namespace
from libs import runner, dataset, stats_eval, utils
from configs import *
from libs.lib_svd import Inpainting_custom
import h5py

from configs.plot_config import plot_dict, basic_plt_setup 

plot_config = dict2namespace(plot_dict)
basic_plt_setup()


def normalize(x, u_max, u_min, v_max, v_min):
    """
    Normalizes the input tensor `x` with shape (num,2, nx, ny) by using min
    and max values from train dataset to range [-1, 1].
    
    Parameters:
    x (np.ndarray or torch.Tensor): Array or tensor with shape (num, 2, h, w).
    u_max (float): Maximum value for the first channel.
    u_min (float): Minimum value for the first channel.
    v_max (float): Maximum value for the second channel.
    v_min (float): Minimum value for the second channel.
    
    Returns:
    np.ndarray or torch.Tensor: Normalized array or tensor with the same shape as `x`, in the range [-1 to 1].
    """
    is_torch = isinstance(x, torch.Tensor)
    device = x.device if is_torch else None

    x = x.clamp(-1, 1) if is_torch else np.clip(x, a_min=-1, a_max=1)

    eps = 1e-9
    center = torch.tensor([u_min, v_min], device=device).reshape((2, 1, 1)) if is_torch else np.array([u_min, v_min]).reshape((2, 1, 1))
    scale = torch.tensor([u_max - u_min, v_max - v_min], device=device).reshape((2, 1, 1)) if is_torch else np.array([u_max - u_min, v_max - v_min]).reshape((2, 1, 1))
    x_scaled = (x - center) / (scale + eps)

    return (2 * x_scaled) - 1

def renormalize(x_norm, u_max, u_min, v_max, v_min):
    """
    Renormalizes the input tensor `x_norm` with shape (num,2, h, w) from the range [-1, 1]
    back to the original range defined by u_max, u_min, v_max, and v_min.
    
    Parameters:
    x_norm (np.ndarray or torch.Tensor): Normalized array or tensor with shape (num,2, h, w) in the range [-1, 1].
    u_max (float): Maximum value for the first channel.
    u_min (float): Minimum value for the first channel.
    v_max (float): Maximum value for the second channel.
    v_min (float): Minimum value for the second channel.
    
    Returns:
    np.ndarray or torch.Tensor: Renormalized array or tensor with the same shape as `x_norm`, in the original range.
    """
    is_torch = isinstance(x_norm, torch.Tensor)
    device = x_norm.device if is_torch else None

    eps = 1e-9  # Small epsilon to avoid division by zero
    center = torch.tensor([u_min, v_min], device=device).reshape((2, 1, 1)) if is_torch else np.array([u_min, v_min]).reshape((2, 1, 1))
    scale = torch.tensor([u_max - u_min, v_max - v_min], device=device).reshape((2, 1, 1)) if is_torch else np.array([u_max - u_min, v_max - v_min]).reshape((2, 1, 1))

    # Scale back to [0, 1] range
    x_rescaled = (x_norm + 1) / 2

    # Shift and scale back to original range
    x_renormalized = x_rescaled * (scale + eps) + center

    return x_renormalized



def get_data(config, datatype=None):
    OneObs = dataset.get_dataset(config)
    #get mins and maxs from train data
    u_min, u_max = OneObs.u_min, OneObs.u_max
    v_min, v_max = OneObs.v_min, OneObs.v_max

    if datatype == "Train":
        data = OneObs.data
        x, y, t = OneObs.x,  OneObs.y, OneObs.t
        
    elif datatype == "Test":
        data, x, y, t = dataset.get_2Dtest_data(config=config)

    if config.model.use_fp16_for_data:
        data = data.astype(np.float16)
        u_min, u_max = u_min.astype(np.float16), u_max.astype(np.float16)
        v_min, v_max = v_min.astype(np.float16), v_max.astype(np.float16)
        x, y, t = x.astype(np.float16), y.astype(np.float16), t.astype(np.float16)

    print(f"Data selected:{datatype} with shape:{data.shape} and data type {data.dtype}", flush=True)

    return data, u_max, u_min, v_max, v_min, x, y ,t

def get_skipped_steps(array, start, stop, step):
    """
    Extracts the elements of an array that are skipped in a slicing operation.

    Parameters:
        array (numpy.ndarray): The input array to slice.
        start (int): The start index for the slicing.
        stop (int): The stop index for the slicing.
        step (int): The step size for the slicing.

    Returns:
        numpy.ndarray: An array containing the skipped steps.
    """
    # Generate all indices in the range
    all_indices = np.arange(start, stop)
    
    # Generate the indices that are included in the slicing
    included_indices = np.arange(start, stop, step)
    
    # Find the indices that are skipped
    skipped_indices = np.setdiff1d(all_indices, included_indices)
    
    # Extract the skipped steps from the array
    skipped_array = array[skipped_indices]
    
    return skipped_array


def create_random_mask(snapshot_shape, percentage):
    # Determine the number of pixels to select
    height, width = snapshot_shape[1], snapshot_shape[2]
    total_pixels = height * width
    num_sensors_to_select = int(total_pixels * (percentage / 100))
    
    # Generate random pixel locations
    all_indices = np.arange(total_pixels)
    selected_indices = np.random.choice(all_indices, num_pixels_to_select, replace=False)
    
    # Create the mask
    mask = np.zeros(total_pixels, dtype=int)
    mask[selected_indices] = 1
    
    # Reshape the mask to the original snapshot shape
    mask = mask.reshape((height, width))
    
    # Replicate the mask for both snapshots
    masks = np.stack([mask, mask])
    
    return masks

def create_segment(seg_tensor=None, x1=None, x2=None, y1=None, y2=None, 
                   subregion_x_size=None, subregion_y_size=None, counter=1):
    assert y1 < y2 and x1 < x2, "Invalid range: x1 should be less than x2 and y1 should be less than y2."
    assert seg_tensor is not None, "seg_tensor should not be None."
    assert subregion_x_size is not None and subregion_y_size is not None, "subregion_x_size and subregion_y_size should not be None."

    # Calculate nx and ny based on the subregion sizes
    nx = math.ceil((x2 - x1) / subregion_x_size)
    ny = math.ceil((y2 - y1) / subregion_y_size)

    print(f"counter: {counter}, subregion_x_size:{subregion_x_size}, subregion_y_size:{subregion_y_size}, nx:{nx}, ny:{ny}")
    
    # Fill each subregion with consecutive integers starting from the given counter
    for i in range(nx):
        for j in range(ny):
            x_start = x1 + i * subregion_x_size
            x_end = min(x_start + subregion_x_size, x2)  # Ensure boundary alignment

            y_start = y1 + j * subregion_y_size
            y_end = min(y_start + subregion_y_size, y2)  # Ensure boundary alignment

            seg_tensor[:, x_start:x_end, y_start:y_end] = counter
            counter += 1
            
    return seg_tensor, counter

def create_segment_legacy(seg_tensor=None, x1=None, x2=None, y1=None, y2=None, nx=None, ny=None, counter=1):
    """Unused variant kept for reference: subregions given by their number (nx, ny) instead of their pixel size."""
    assert y1 < y2 and x1 < x2, "Invalid range: x1 should be less than x2 and y1 should be less than y2."
    assert seg_tensor is not None, "seg_tensor should not be None."
    assert nx is not None and ny is not None, "nx and ny should not be None."

    # Calculate the size of each subregion
    subregion_x_size = math.ceil((x2 - x1) / nx)
    subregion_y_size = math.ceil((y2 - y1) / ny)
 
    print(f"counter: {counter}, subregion_x_size:{subregion_x_size}, subregion_y_size:{subregion_y_size}")
    # Fill each subregion with consecutive integers starting from the given counter
    for i in range(nx):
        for j in range(ny):
            x_start = x1 + i * subregion_x_size
            x_end = min(x_start + subregion_x_size, x2)  # Ensure boundary alignment

            y_start = y1 + j * subregion_y_size
            y_end = min(y_start + subregion_y_size, y2)  # Ensure boundary alignment

            seg_tensor[:, x_start:x_end, y_start:y_end] = counter
            counter += 1
            
    return seg_tensor, counter


def create_mask(data=None, mask=None, ds_ratio=None, mask_type=None, segmentation=False):
    """
    It is important to calculate mask with respect to building height (in % terms of buidling, h)
    """
    
    if data is None or mask is None or ds_ratio is None or mask_type is None:
        raise ValueError("Data, mask, ds_ratio, and mask_type must be provided.")

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Define obstacle location in units
    x_obs_i, x_obs_f, y_obs_i, y_obs_f = -0.125, 0.125, 0, 1

    # Convert obstacle location to pixels
    x_obs_pix_i, y_obs_pix_i, _ = utils.convert_unit2pixel(x=x_obs_i, y=y_obs_i, ds_ratio=ds_ratio)
    x_obs_pix_f, y_obs_pix_f, _ = utils.convert_unit2pixel(x=x_obs_f, y=y_obs_f, ds_ratio=ds_ratio)

    print(f"Obstacle is located at pixels: ({x_obs_pix_i}, {y_obs_pix_i}) with (width, height): ({x_obs_pix_f - x_obs_pix_i}, {y_obs_pix_f - y_obs_pix_i})", flush=True)

    # Convert data to torch tensor
    data_torch = torch.from_numpy(data).to('cuda')
    mask_tensor = torch.zeros_like(data_torch[0])
    seg_tensor = None

    # Create mask based on mask type
    if mask_type == "from_ground":
        y_unmask_grnd = int((mask / 100) * data_torch.shape[-1])
        mask_tensor[:, :, :y_unmask_grnd] = 1
        print(f"Number of pixel rows selected from the ground: {y_unmask_grnd}")

    elif mask_type == "random":
        mask_tensor = torch.from_numpy(create_random_mask(data[0].shape, mask)).to('cuda')

    elif mask_type in "from_ground_and_wall":
        x_unmask_build = x_obs_pix_f + int((mask / 100) * y_obs_pix_f) #open region in x near building
        y_unmask_build = y_obs_pix_f + int((mask / 100) * y_obs_pix_f) #open region in y near building
        y_unmask_grnd = int((mask / 100) * y_obs_pix_f)              #open region in y above ground

        print(f"{y_obs_pix_i}, {y_obs_pix_f}, {y_unmask_build}")
        print(f"{x_obs_pix_i}, {x_obs_pix_f}, {x_unmask_build}")

        mask_tensor[:, x_obs_pix_f:x_unmask_build, :y_unmask_build]     = 1
        mask_tensor[:, x_obs_pix_i:x_obs_pix_f, y_obs_f:y_unmask_build] = 1
        mask_tensor[:, x_obs_pix_f:, :y_unmask_grnd]                  = 1
        print(f"Number of pixel rows selected from the ground: {y_unmask_grnd}")

        # Ensure obstacle area is masked
        mask_tensor[:, x_obs_pix_i:x_obs_pix_f, y_obs_pix_i:y_obs_pix_f] = 1    
        
        if segmentation:
            seg_tensor = torch.zeros_like(mask_tensor)

            #nx1, nx2, nx3 = 20, 10, 116
            #ny1, ny2, ny3 = 5, 15, 5
            subregion_x_size, subregion_y_size =5,1   #2, 1

            seg_tensor, counter = create_segment(seg_tensor=seg_tensor, x1 = x_obs_pix_i, x2 = x_unmask_build, y1=y_obs_pix_f, y2=y_unmask_build, subregion_x_size=subregion_x_size, subregion_y_size=subregion_y_size, counter=1)
            seg_tensor, counter = create_segment(seg_tensor=seg_tensor, x1 = x_obs_pix_f, x2 = x_unmask_build, y1=y_unmask_grnd, y2=y_obs_pix_f, subregion_x_size=subregion_x_size, subregion_y_size=subregion_y_size, counter=counter)
            seg_tensor, counter = create_segment(seg_tensor=seg_tensor, x1 = x_obs_pix_f, x2 = seg_tensor.shape[-2], y1=y_obs_pix_i, y2=y_unmask_grnd, subregion_x_size=subregion_x_size, subregion_y_size=subregion_y_size, counter=counter)
    
    print(f"Mask tensor shape: {mask_tensor.shape} and type {mask_tensor.dtype}")
    print(f"gtruth shape: {data_torch.shape} and type {data_torch.dtype}")
    
    assert mask_tensor.shape == data_torch.shape[1:]

    return mask_tensor, data_torch, seg_tensor


def create_random_mask_within_ground_and_wall(seg_tensor=None, select_random_sensors=None, seed=0):
    """
    Creates a random mask with `select_random_sensors` values from `seg_tensor` set to 1, based on unique values in `seg_tensor`.
    
    Parameters:
    - seg_tensor (torch.Tensor): The segmentation tensor to modify.
    - select_random_sensors (int): Number of unique values to randomly select.
    
    Returns:
    - torch.Tensor: Mask with 1s for selected values and 0s elsewhere.
    """
    if seg_tensor is None or select_random_sensors is None:
        raise ValueError("seg_tensor and select_random_sensors must be provided.")
    
    # Get unique non-zero values in seg_tensor
    unique_values, counts = torch.unique(seg_tensor, return_counts=True)
    unique_values = unique_values[unique_values != 0.0] #Remove the background
    
    print(f"Total available sensors:{len(unique_values)}")
    if unique_values.size(0) < select_random_sensors:
        raise ValueError("Number of unique values to select exceeds available unique values.")
    
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Use the generator for random operations
    random_indices = torch.randperm(unique_values.size(0), generator=generator)[:select_random_sensors]
    selected_values = unique_values[random_indices]
    
    # Create a mask tensor
    mask_tensor = torch.zeros_like(seg_tensor, dtype=torch.float32)
    
    # Vectorized masking: Check where seg_tensor matches any of the selected values
    mask_tensor = torch.isin(seg_tensor, selected_values)#.to(dtype=torch.float32)

    return mask_tensor

def plot_mask(mask_tensor=None, obs=None, mask=None, 
                 x_axis=None, y_axis=None, mask_type=None, 
                cmap=None, segmentation=False, vmin=None, vmax=None,
                out_pdf=False, out_png=False, out_show=False):
    """
    Plots the given mask tensor along with the obstacle and sensor data if provided.

    Parameters:
    mask_tensor (torch.Tensor): The mask tensor to be plotted.
    obs (torch.Tensor, optional): The sensor data tensor. Defaults to None.
    mask (float, optional): The mask percentage. Defaults to None.
    x_axis (numpy.ndarray): The x-axis values.
    y_axis (numpy.ndarray): The y-axis values.
    mask_type (str): The type of mask applied.

    Raises:
    ValueError: If mask_tensor, x_axis, or y_axis are None.
    """
    
    if mask_tensor is None:
        raise ValueError("mask_tensor cannot be None")

    if x_axis is None or y_axis is None:
        raise ValueError("x_axis and y_axis cannot be None")

    # Define the extent based on x_axis and y_axis
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    if isinstance(mask_tensor, torch.Tensor):
        mask_tensor = mask_tensor.cpu().numpy()

    try:
        if obs is not None:
            sensor = obs.cpu().numpy()
            sensor = sensor.reshape((-1, 2, mask_tensor.shape[-2], mask_tensor.shape[-1]))

            #print(f"Sensor input shape: {sensor.shape}, dtype: {sensor.dtype}")
            
            # Plot the sensor data
            im = ax.imshow(sensor[0, 0].T, cmap=cmap, extent=extent, origin='lower', aspect='auto')
            ax.set_title("Sensor Inputs (With Masks)")
            fig.colorbar(im, ax=ax, orientation='vertical')

        elif segmentation:
            
            labeled_array = np.zeros_like(mask_tensor[0])
            non_zero_indices = np.where(mask_tensor[0] != 0)
            labeled_array[non_zero_indices] = (mask_tensor[0][non_zero_indices] % 2) + 1

            cmap = ListedColormap(['white', 'black', 'lightgray'])
            
            im = ax.imshow(labeled_array[: , :].T, extent=extent, cmap=cmap, interpolation='none', origin='lower', aspect='auto')
            

        else:
            #for i in range(mask_tensor[0].shape[0]):
            #    for j in range(mask_tensor[0].shape[1]):
            #        value = mask_tensor[0][i, j]
            #        if value != 0:  # Skip zero values if desired
            #            ax.text(j, i, f'{value}', ha='center', va='center', color='red', fontsize=25)
            
            if vmin is None or vmax is None:
                vmin, vmax = np.min(mask_tensor), np.max(mask_tensor)
            
            if cmap == None:
                cmap = ListedColormap(['white', 'lightgray'])
            im = ax.imshow(mask_tensor[0].T, cmap=cmap, extent=extent, interpolation='nearest', vmin=vmin, vmax= vmax, origin='lower', aspect='auto')
            ax.set_title("Mask Tensor")
            fig.colorbar(im, ax=ax, orientation='vertical')

        # Common plot settings
        ax.set_xlabel(plot_config.axes.x_label)
        ax.set_ylabel(plot_config.axes.y_label)

        if cmap == None:
            utils.add_obstacle_patch(ax, color='gray')
        else:
            utils.add_obstacle_patch(ax)
        
        if out_pdf:
            plt.savefig(f'mask-ds1-{mask_type}-{mask}.pdf', dpi=plot_config.figure.dpi)
        if out_png:
            plt.savefig(f'mask-ds1-{mask_type}-{mask}.png', dpi=plot_config.figure.dpi)
        if out_show:    
            plt.show()  

    finally:
        plt.close(fig)

def plot_sensor_importance(
    sensor_importance_array=None,
    x_axis=None,
    y_axis=None,
    plot_name=None,
    num_sensors=None,
    out_show=False,
    out_pdf=False,
    out_png=False
):
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    fontsize = 40
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    # Define discrete levels and custom labels for the colorbar
    levels = list(range(len(num_sensors)+1))  # Boundaries for discrete color mapping
    custom_labels = num_sensors 
    tick_positions = np.arange(len(num_sensors)) + 0.5

    vmin, vmax = np.min(sensor_importance_array), np.max(sensor_importance_array)

    # Create a discrete colormap
    cmap = plt.get_cmap("inferno")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Plot the sensor importance array
    im = ax.imshow(
        sensor_importance_array[0].T,
        cmap=cmap,
        extent=extent,
        norm=norm,
        origin="lower",
        aspect="auto",
    )
    #ax.set_title(plot_name)

    ax.set_xlabel(plot_config.axes.x_label, fontsize=fontsize)
    ax.set_ylabel(plot_config.axes.y_label, fontsize=fontsize)

    ax.tick_params(axis='both', labelsize=fontsize)

    # Add obstacle patch (customize as needed)
    utils.add_obstacle_patch(ax, obs_color="lightgrey") 

    # Add the discrete colorbar with custom labels
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", ticks=tick_positions, spacing="uniform")
    cbar.ax.set_yticklabels(custom_labels)  # Set the custom labels
    cbar.ax.tick_params(labelsize=fontsize-10) 
    cbar.ax.yaxis.set_label_position('right')

    cbar.set_label(rf"\% cummulative pixels", fontsize=fontsize-10, labelpad=5, rotation=90)

    # Save or show the plot
    if out_pdf:
        plt.savefig(f"Sensor Importance-{plot_name}.pdf", dpi=300)
    if out_png:
        plt.savefig(f"Sensor Importance-{plot_name}.png", dpi=300)
    if out_show:
        plt.show()


def reduce_sensor_clusters2pixels(mask_tensor=None, block_width=3, block_height=2, pixel_loc="Top-left"):
    """
    Reduce clusters of sensor locations (represented by 1's) to a single pixel per sub-block of size block_width x block_height.
    
    Args:
    mask_tensor (torch.Tensor or np.ndarray): Input mask where 1's represent sensor locations.
    block_width (int): Width of the sub-block (default: 3).
    block_height (int): Height of the sub-block (default: 2).
    pixel_loc (str): Location of the pixel to select within each block (default: "Top-left").
    
    Returns:
    torch.Tensor: New mask with one pixel selected per block of size block_width x block_height.
    """
    from scipy.ndimage import label

    
    # Convert torch.Tensor to numpy if necessary
    if isinstance(mask_tensor, torch.Tensor):
        mask_tensor = mask_tensor.cpu().numpy()

    # Label connected components (clusters)
    labeled_array, num_features = label(mask_tensor)

    # Create a new mask to store the selected pixels
    new_mask_tensor = np.zeros_like(mask_tensor)

    # Loop over each labeled cluster
    for i in range(1, num_features + 1):
        # Find the indices of the current cluster
        cluster_indices = np.argwhere(labeled_array[0] == i)
        
        #Get sensor location for standalone clusters first,
        #(This might duplicate sensor locations for larger clusters)
        first_pixel = cluster_indices[0]
        new_mask_tensor[0, first_pixel[0], first_pixel[1]] = 1

        if cluster_indices.size == 0:
            continue

        # Get min and max row and column indices of the cluster
        min_row, max_row = cluster_indices[:, 0].min(), cluster_indices[:, 0].max()
        min_col, max_col = cluster_indices[:, 1].min(), cluster_indices[:, 1].max()
        
        # Get the length of the cluster in x and y directions
        cluster_width = max_col - min_col + 1
        cluster_height = max_row - min_row + 1
        
        # Number of blocks in x and y directions
        blocks_in_x = cluster_width // block_width
        blocks_in_y = cluster_height // block_height

        # Iterate through each block
        for y in range(blocks_in_y):
            for x in range(blocks_in_x):
                # Determine the starting pixel of each block (top-left by default)
                
                if pixel_loc == "Top-left":
                    start_row = min_row + y * 2
                    start_col = min_col + x * 3
                    
                    # Select the top-left pixel of this block
                    new_mask_tensor[0,start_row, start_col] = 1
                else:
                    raise NotImplementedError

    return torch.from_numpy(new_mask_tensor)
