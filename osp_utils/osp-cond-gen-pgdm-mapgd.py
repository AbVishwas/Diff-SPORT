import numpy as np
import torch
import sys
import os
import h5py
from pathlib import Path

# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from libs import runner, dataset, stats_eval, cond_gen, utils, shap_eval
from configs import *
from libs.lib_svd import Inpainting_custom
from libs.cond_gen import plot_mask


CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
              }


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

configname = sys.argv[1]
osp_method = sys.argv[2]
threshold = float(sys.argv[3]) if sys.argv[3] != "None" else None

start = int(sys.argv[4])
stop = int(sys.argv[5])
step = int(sys.argv[6])
datatype = sys.argv[7]  
strategy= sys.argv[8]
gen_seed= int(sys.argv[9])
mask=30

version          = 'v5.5-tdiff20-gd50-snaps-25000-step50'  
mask_file_name   = f"shap_v5.5_mask30-thresh{threshold}-strategy-{strategy}.npz"
qr_mask_filename = f"qr_mask-start-0-stop-25000-step-50-num_sensors{threshold}-bs-500.npz" 
                     
print(f"selected config ={configname}")
config_dict = CONFIG_DICT[configname]
config = runner.dict2namespace(config_dict)

# Load data
data, u_max, u_min, v_max, v_min, x, y ,t = cond_gen.get_data(config, datatype= datatype.removesuffix("-interpolation"))

if datatype == "Train" or datatype =="Test":
    data  = data[start:stop:step] 
    t = t[start:stop:step]
    
elif datatype == "Train-interpolation":
    data = cond_gen.get_skipped_steps(data, start, stop, step) #Get data to check interpolation
    t =  cond_gen.get_skipped_steps(t, start, stop, step)
snaps = (stop - start)//step
print(f"Generating samples for {osp_method}-Threshold:{threshold} with {config.sampling.T_sub} ", flush =True)

train_mean = shap_eval.get_train_mean(config=config)
#Load Masks
osp_dir = config.eval.osp_dir

if osp_method == "shap":
    osp_method_dir = osp_dir + "shap/"
    results_dir = osp_method_dir + f"results/{configname}/" + version + "/"
    gen_dir = osp_method_dir + f"generated_samples/{configname}/" + version + "/" + strategy + "/"
    threshold_mask_path = osp_method_dir + f"threshold-masks/" + version + "/" + strategy + "/" + mask_file_name

    # Create directories if they do not exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    #Load osp mask
    mask_file = np.load(threshold_mask_path)
    mask_tensor = mask_file["shap_mask"]
    mask = mask_file["mask"]
    mask_type = "shap_" + version

    print(f"Generating with full shap masks: {mask_type}")

elif osp_method == "shap-reduced-mask":
    osp_method_dir = osp_dir + "shap/"
    results_dir = osp_method_dir + f"results/{configname}/" + version + "/"
    gen_dir = osp_method_dir + f"generated_samples/{configname}/" + version + "/" + strategy + "/"
    threshold_mask_path = osp_method_dir + f"threshold-masks/" + version + "/" + strategy + "/" + mask_file_name

    # Create directories if they do not exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    #Load osp mask
    mask_file = np.load(threshold_mask_path)
    mask_tensor = mask_file["reduced_shap_mask"]
    mask = mask_file["mask"]
    mask_type = "reduced_" + version

    print(f"Generating with reduced shap masks: {mask_type}")


elif osp_method == "qr":
    osp_method_dir = osp_dir + "qr-pivoting/"
    results_dir = osp_method_dir + f"results/"
    gen_dir = osp_method_dir + f"generated_samples/"
    
    threshold_mask_path=osp_method_dir + f"qr_masks/" + qr_mask_filename 

    # Create directories if they do not exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    #Load osp mask
    mask_file = np.load(threshold_mask_path)
    mask_tensor = mask_file["qr_mask"]
    mask        = mask_file["mask"]
    mask_type   = "qr"
    strategy = "None"

    print(f"Generating with qr masks for num_sensors: {int(threshold)}")


if not isinstance(mask_tensor, torch.Tensor):
    mask_tensor = torch.from_numpy(mask_tensor).to('cuda')


#normalize test data using mins and maxs from train
data_norm = cond_gen.normalize(data, u_max, u_min, v_max, v_min)

if not isinstance(data_norm, torch.Tensor):
    data_norm = torch.from_numpy(data_norm).to('cuda')

H = Inpainting_custom(mask_tensor=mask_tensor, device="cuda")
obs = H.H(data_norm)

print(f"Initiating Model...", flush =True)
config, ddpm_model, schedule = runner.create_model_and_initialize_ddpm(config_dict)
ddpm_model = runner.load_model(config, ddpm_model)

T_sub = config.sampling.T_sub
T_sub_mapgd = config.sampling.T_sub_mapgd
gditer_mapgd = config.sampling.gditer_mapgd

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
       #We dont generate with seed for ddrm and pgdm, only for mapgd, since it is more accurate
       x_recovered_ddrm = ddpm_model.reverse_diffusion_ddrm(data_norm_batch.shape, obs_batch, sigma_y=0, H=H, T_ddrm=T_sub, cuda=True, onlymean=False, adaptive=False, y_c=None, class_cond=False)

       pgdm_time_start = utils.get_current_time()
       x_recovered_pgdm = ddpm_model.reverse_diffusion_pgdm(data_norm_batch.shape, obs_batch, sigma_y=0, H=H, T_ddrm=T_sub, cuda=True, onlymean=False, adaptive=False, y_c=None, class_cond=False)
       pgdm_time_end = utils.get_current_time()

       mapgd_time_start = utils.get_current_time()
       x_recovered_mapgd = ddpm_model.reverse_diffusion_gd(data_norm_batch.shape, obs_batch, sigma_y=0, H=H, T_ddrm=T_sub_mapgd, cuda=True, onlymean=False,  karras=False, adaptive=False, y_c=None, class_cond=False, gditer=gditer_mapgd, seed=gen_seed)
       mapgd_time_end = utils.get_current_time()

       x_recovered_cpu_ddrm = x_recovered_ddrm.cpu().numpy()
       x_recovered_cpu_pgdm = x_recovered_pgdm.cpu().numpy()
       x_recovered_cpu_mapgd = x_recovered_mapgd.cpu().numpy()

       pred_images_ddrm.append(x_recovered_cpu_ddrm)
       pred_images_pgdm.append(x_recovered_cpu_pgdm)
       pred_images_mapgd.append(x_recovered_cpu_mapgd)

       batch_end_time = utils.get_current_time() 
       print(f"batch {i} in finished in {utils.get_time_difference(batch_start_time, batch_end_time)}", flush=True)
       print(f"batch {i} in mapgd time {utils.get_time_difference(mapgd_time_start, mapgd_time_end)}", flush=True)


end_time = utils.get_current_time()
print(f"Generated {snaps} samples in {utils.get_time_difference(start_time, end_time)} with gen_seed: {gen_seed}", flush=True)

pred_ddrm = np.concatenate(pred_images_ddrm, axis=0)
pred_pgdm = np.concatenate(pred_images_pgdm, axis=0)
pred_mapgd = np.concatenate(pred_images_mapgd, axis=0)

pred_ddrm_renorm = cond_gen.renormalize(pred_ddrm, u_max, u_min, v_max, v_min)
pred_pgdm_renorm = cond_gen.renormalize(pred_pgdm, u_max, u_min, v_max, v_min)
pred_mapgd_renorm = cond_gen.renormalize(pred_mapgd, u_max, u_min, v_max, v_min)

filename_hdf5 = f"osp-threshold-{threshold}-cond_gen-{mask_type}-strategy-{strategy}-seed{gen_seed}-M{int(mask)}%-T_sub-{T_sub}-T_sub_mapgd-{T_sub_mapgd}-gditer_mapgd-{gditer_mapgd}-start-{start}-stop-{stop}-step-{step}-{datatype}.h5"

with h5py.File(gen_dir + filename_hdf5, 'w') as hf:
    hf.create_dataset("data", data=data)
    hf.create_dataset("ddrm", data=pred_ddrm_renorm)
    hf.create_dataset("pgdm", data=pred_pgdm_renorm)
    hf.create_dataset("mapgd", data=pred_mapgd_renorm)
    hf.create_dataset("mask_tensor", data = mask_tensor.cpu().numpy())
    hf.create_dataset("osp_method", data=osp_method)
    hf.create_dataset("threshold", data=threshold)
    hf.create_dataset("x", data = x )
    hf.create_dataset("y", data = y )
    hf.create_dataset("t", data = t )
    hf.create_dataset("T_sub", data=T_sub)
    hf.create_dataset("T_sub_mapgd", data=T_sub_mapgd)
    hf.create_dataset("gditer", data=gditer_mapgd)

print(f"MAPGD conditional generation with osp mask completed!!", flush =True)
