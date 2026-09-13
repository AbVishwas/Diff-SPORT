import numpy as np
import torch
import sys
import os

sys.path.append('../../')

import libs.shap.shap as shap
from libs import runner, dataset, stats_eval, utils, cond_gen, shap_eval
from configs import *
import h5py
from libs.shap_osp import shap_osp, get_shap_features
from libs.cond_gen import plot_mask, create_mask


CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
            }

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices available.")


config_name = sys.argv[1]
mask_type   = sys.argv[2]
mask        = float(sys.argv[3]) #to be unmasked, from bottom edge
start        = int(sys.argv[4])
stop        = int(sys.argv[5])
step        = int(sys.argv[6])
ncoalitions = int(sys.argv[7])
select_data = sys.argv[8]


print(f"selected config ={config_name}")
config_dict = CONFIG_DICT[config_name]

config, ddpm_model, schedule = runner.create_model_and_initialize_ddpm(config_dict)
ddpm_model = runner.load_model(config, ddpm_model)

T_sub = config.sampling.T_sub
T_sub_mapgd = config.sampling.T_sub_mapgd
gditer_mapgd = config.sampling.gditer_mapgd

#Load data
gtruth, u_max, u_min, v_max, v_min, x, y ,t = cond_gen.get_data(config, datatype= "Train") #Test

u_mag      = utils.get_velocity_magnitude(data=gtruth)
u_mag_mean = np.mean(u_mag, axis=0)
train_mean = shap_eval.get_train_mean(config)

if select_data == "sequential":
    gtruth     = gtruth[start:stop:step]
    u_mag_snap = u_mag[start:stop:step]

elif select_data == "random":
    gtruth = gtruth[start:stop]
    u_mag_snap = u_mag[start:stop]
    snaps = (stop - start)//step
    np.random.seed(seed)
    random_indices = np.random.choice(gtruth.shape[0], size=snaps, replace=False)
    gtruth = gtruth[random_indices]
    u_mag_snap = u_mag_snap[random_indices]

print(f"After filtering ground truth data shape-{gtruth.shape} from {start} to {stop}",flush=True)

mask_tensor_0, gtruth_torch, _            = create_mask(data=gtruth, mask=0, ds_ratio=config.dataset.ds_ratio, mask_type=mask_type, segmentation=False)
mask_tensor_100, gtruth_torch, seg_tensor = create_mask(data=gtruth, mask=mask, ds_ratio=config.dataset.ds_ratio, mask_type=mask_type, segmentation=True)

plot_mask(mask_tensor=mask_tensor_0.cpu().numpy(), mask=mask, x_axis=x, y_axis=y, mask_type="zero_mask" + mask_type , out_pdf=True)
plot_mask(mask_tensor=mask_tensor_100.cpu().numpy(), mask=mask, x_axis=x, y_axis=y, mask_type=mask_type , out_pdf=True)
plot_mask(mask_tensor=seg_tensor.cpu().numpy(), mask=mask, x_axis=x, y_axis=y, mask_type="segmentation_"+mask_type, segmentation=True, out_pdf=True)

unique_values, counts = torch.unique(seg_tensor, return_counts=True)
unique_values = unique_values[torch.where(unique_values != 0.0)] #We remove seg_tensor value 0, which represents background
nmax2= len(unique_values) 

zzeros = np.zeros((1,nmax2)) 
zshap = np.ones((1,nmax2)) 

#Data is not normalized before passing to this class
shap_inst = shap_osp(schedule, ddpm_model, gtruth_tensor = gtruth_torch, 
                        sigma_y=0, T_sub=T_sub, weightedloss=True, cuda=True, 
                        onlymean=False, adaptive=False, y_c=None, class_cond=False,
                        zero_mask = mask_tensor_0, full_mask= mask_tensor_100, 
                        seg_tensor = seg_tensor, u_max=u_max, u_min=u_min, 
                        v_max=v_max, v_min=v_min,  x_axis =x, y_axis=y, time=t, unique_segments= unique_values,
                        zzero=zzeros, zshap=zshap, train_mean=train_mean, config=config, T_sub_mapgd=T_sub_mapgd, gditer_mapgd = gditer_mapgd)

print(f"unique_values in unique_values:{unique_values.min()}:{unique_values.max()}, min:max of seg_tensor:{seg_tensor.min()}:{seg_tensor.max()}, Number of unique values:{nmax2}",flush=True)

explainer = shap.KernelExplainer(shap_inst.model_function_mask_batches,zzeros)
shaps = explainer.shap_values(zshap,nsamples=ncoalitions) #"auto")

case_name =f"shap-M{mask}%-{mask_type}-t_sub-{T_sub}-T_sub_mapgd-{T_sub_mapgd}-gditer-{gditer_mapgd}-start{start}-stop{stop}-step{step}-ncoali{ncoalitions}"
file_path = f"./results/shaply-values/{case_name}.npz"

np.savez(file_path, 
         shaps=shaps, 
         seg_tensor=seg_tensor.cpu().numpy(), 
         mask=mask, 
         mask_type=mask_type, 
         x=x, 
         y=y, 
         t=t, 
         T_sub=T_sub,
         zzeros=zzeros,
         zshap=zshap)

print(f"Shap values saved!!!")
