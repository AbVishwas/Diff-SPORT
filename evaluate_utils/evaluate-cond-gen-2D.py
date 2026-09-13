import sys
import os
import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# Ensure we can import local packages regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from libs    import runner, utils, inst_eval
from configs import *  


CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
            }

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

configname = sys.argv[1]
mask_type = sys.argv[2]
mask      = int(sys.argv[3])
snaps     = int(sys.argv[4])
num       = int(sys.argv[5])
results_dir = sys.argv[6].lstrip('/')  # ensure relative so we don't write to /results

print(f"configname={configname}")
config_dict = CONFIG_DICT[configname]

config = runner.dict2namespace(config_dict)

T_sub = config.sampling.T_sub
T_sub_mapgd = config.sampling.T_sub_mapgd
gditer_mapgd = config.sampling.gditer_mapgd

cond_inf_dir = REPO_ROOT / config.eval.cond_gen_dir
save_dir   = REPO_ROOT / config.eval.eval_dir / results_dir / configname 

filename_hdf5 = f"cond_gen-{mask_type}-{int(mask)}%-ckpt-{config.sampling.inf_epochs}-T_sub_mapgd-{T_sub_mapgd}-gditer-{gditer_mapgd}-{snaps/1000}k-mask_seed-0-gen_seed-0-num_sensors-None.h5"
print(f"filename: {filename_hdf5}")

file = h5py.File(str(cond_inf_dir / filename_hdf5), 'r')

gtruth = file["data"][:]
pred_ddrm = file["ddrm"][:]
pred_pgdm = file["pgdm"][:]
pred_mapgd = file["mapgd"][:]
mask_tensor = file["mask_tensor"][:]
x_axis = file["x"][:]
y_axis = file["y"][:]
time = file["t"][:]

inst_evaluator = inst_eval.InstantaneousEvaluation(gtruth= gtruth[:], pred1= pred_mapgd[:], pred2=pred_pgdm[:], x_axis= x_axis, y_axis=y_axis, 
                        time=time[:], input_data_type="2D", config = config)


os.makedirs(str(save_dir), exist_ok=True)
print(f"Saving pdf at {save_dir}")

pdf = PdfPages(str(save_dir / f'cond_eval-cond_gen-{mask_type}-{int(mask)}%-ckpt-{config.sampling.inf_epochs}-T_sub-{T_sub}-gditer-{gditer_mapgd}-{snaps/1000}k-num-{num}.pdf'))

random_indices = utils.select_random(gtruth[:], num_elements=num, seed=0, only_indices=True)
pdf = inst_evaluator.main(random_indices = random_indices, mask = mask, ds_ratio=1, pdf=pdf)

pdf.close()
