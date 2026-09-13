import numpy as np
import torch
import sys
import os
from pathlib import Path

# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from libs import runner, cond_gen, utils
from configs import *
from libs.lib_svd import Inpainting_custom
import h5py


CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
            }

config_name = sys.argv[1]
mask_type   = sys.argv[2]
mask        = float(sys.argv[3]) #to be unmasked, from bottom edge
snaps       = int(sys.argv[4])   #Number of measurements snaps given, sequential
gen_seed    = int(sys.argv[5])
mask_seed   = int(sys.argv[6])

try:
    select_random_sensors = int(sys.argv[7]) if sys.argv[7] else None
except (IndexError, ValueError):
    select_random_sensors = None

print(f"selected config ={config_name}")
config_dict = CONFIG_DICT[config_name]

#Get Model
config, ddpm_model, schedule = runner.create_model_and_initialize_ddpm(config_dict)
ddpm_model = runner.load_model(config, ddpm_model)

T_sub = config.sampling.T_sub
T_sub_mapgd = config.sampling.T_sub_mapgd
gditer_mapgd = config.sampling.gditer_mapgd

model_precision_type = "float16" if config.model.use_fp16 else "float32"
data_precision_type = "float16" if config.model.use_fp16_for_data else "float32"

#Load data
data, u_max, u_min, v_max, v_min, x, y ,t = cond_gen.get_data(config, datatype= "Test") #Test
data = data[:snaps]
t = t[:snaps]

#normalize test data using mins and maxs from train
data_norm = cond_gen.normalize(data, u_max, u_min, v_max, v_min)

print(f"Creating Masks: {mask_type}, mask:{mask}%", flush=True)

#Create mask (data norm in torch)
mask_tensor, data_norm, seg_tensor = cond_gen.create_mask( data = data_norm, mask = mask, ds_ratio=config.dataset.ds_ratio, mask_type=mask_type, segmentation=True)

if select_random_sensors:
    mask_tensor = cond_gen.create_random_mask_within_ground_and_wall(seg_tensor=seg_tensor, select_random_sensors=select_random_sensors, seed=mask_seed)
    num_pixels = len(torch.nonzero(mask_tensor[0]!=0))
    cond_gen.plot_mask(mask_tensor=mask_tensor, obs=None, mask = mask, x_axis=x, y_axis=y, mask_type=mask_type+f"-mask_seed-{mask_seed}-num_pixels-{num_pixels}", out_pdf=True)
    print(f"Number of sensors selected: {select_random_sensors} corresponding to {num_pixels} pixels")


H = Inpainting_custom(mask_tensor=mask_tensor, device="cuda")

# Now use H to generate y (i.e the observations) from x
obs = H.H(data_norm)

cond_gen.plot_mask(mask_tensor=seg_tensor, obs=None, mask = mask, x_axis=x, y_axis=y, mask_type=mask_type, segmentation=True, out_pdf=True)


print(f"Initiating Model...", flush=True)
# use ddrm and pgdm
pred_images_ddrm = []
pred_images_pgdm = []
pred_images_mapgd = []
batch_size = config.sampling.batch_size
num_batch = int(data_norm.shape[0]/batch_size)

start_time = utils.get_current_time()
print(f"Starting to generate samples @ {start_time}", flush=True)

for i in range(num_batch):
       
       batch_start_time = utils.get_current_time()

       obs_batch        = obs[i*batch_size:(i+1)*batch_size].cuda()
       data_norm_batch  = data_norm[i*batch_size:(i+1)*batch_size].cuda()

       x_recovered_ddrm = ddpm_model.reverse_diffusion_ddrm(data_norm_batch.shape, obs_batch, sigma_y=0, H=H, T_ddrm=T_sub, cuda=True, onlymean=False, adaptive=False, y_c=None, class_cond=False)

       pgdm_time_start = utils.get_current_time()
       x_recovered_pgdm = ddpm_model.reverse_diffusion_pgdm(data_norm_batch.shape, obs_batch, sigma_y=0, H=H, T_ddrm=T_sub, cuda=True, onlymean=False, adaptive=False, y_c=None, class_cond=False)
       pgdm_time_end = utils.get_current_time()

       mapgd_time_start = utils.get_current_time()
       x_recovered_mapgd = ddpm_model.reverse_diffusion_gd(data_norm_batch.shape, obs_batch, sigma_y=0, H=H, T_ddrm=T_sub_mapgd, cuda=True, onlymean=False,  karras=False, adaptive=False, y_c=None, class_cond=False, gditer=gditer_mapgd,  seed=gen_seed)
       mapgd_time_end = utils.get_current_time()

       x_recovered_cpu_ddrm = x_recovered_ddrm.cpu().numpy()
       x_recovered_cpu_pgdm = x_recovered_pgdm.cpu().numpy()
       x_recovered_cpu_mapgd = x_recovered_mapgd.cpu().numpy()

       pred_images_ddrm.append(x_recovered_cpu_ddrm)
       pred_images_pgdm.append(x_recovered_cpu_pgdm)
       pred_images_mapgd.append(x_recovered_cpu_mapgd)

       batch_end_time = utils.get_current_time() 
       print(f"batch {i} in finished in {utils.get_time_difference(batch_start_time, batch_end_time)}", flush=True)
       print(f"batch {i} in pgdm time {utils.get_time_difference(pgdm_time_start, pgdm_time_end)}, mapgd time {utils.get_time_difference(mapgd_time_start, mapgd_time_end)}", flush=True)

end_time = utils.get_current_time()
print(f"Generated {snaps} samples in {utils.get_time_difference(start_time, end_time)}", flush=True)

pred_ddrm = np.concatenate(pred_images_ddrm, axis=0)
pred_pgdm = np.concatenate(pred_images_pgdm, axis=0)
pred_mapgd = np.concatenate(pred_images_mapgd, axis=0)

pred_ddrm_renorm = cond_gen.renormalize(pred_ddrm, u_max, u_min, v_max, v_min)
pred_pgdm_renorm = cond_gen.renormalize(pred_pgdm, u_max, u_min, v_max, v_min)
pred_mapgd_renorm = cond_gen.renormalize(pred_mapgd, u_max, u_min, v_max, v_min)

# Filename for saving the HDF5 file
out_dir = REPO_ROOT / 'inference_utils' / 'conditional_generation' / 'generated_samples'
os.makedirs(str(out_dir), exist_ok=True)

filename_hdf5 = f"cond_gen-{mask_type}-{int(mask)}%-ckpt-{config.sampling.inf_epochs}-T_sub_mapgd-{T_sub_mapgd}-gditer-{gditer_mapgd}-{snaps/1000}k-mask_seed-{mask_seed}-gen_seed-{gen_seed}-num_sensors-{select_random_sensors}.h5"
with h5py.File(str(out_dir / filename_hdf5), 'w') as hf:
    hf.create_dataset("data", data=data)
    hf.create_dataset("ddrm", data=pred_ddrm_renorm)
    hf.create_dataset("pgdm", data=pred_pgdm_renorm)
    hf.create_dataset("mapgd", data=pred_mapgd_renorm)
    hf.create_dataset("mask_tensor", data = mask_tensor.cpu().numpy())
    hf.create_dataset("x", data = x )
    hf.create_dataset("y", data = y )
    hf.create_dataset("t", data = t )
    hf.create_dataset("model_precision_type", data=model_precision_type, dtype='S1')
    hf.create_dataset("data_precision_type", data=data_precision_type, dtype='S1')

print(f"Conditional generation with DDRM and PGDM completed!!")
