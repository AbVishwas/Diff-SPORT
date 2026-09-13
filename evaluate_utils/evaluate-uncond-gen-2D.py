import sys
import os
from pathlib import Path
import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
# Ensure imports work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from libs    import runner, dataset, stats_eval, utils, cond_gen
from configs import *  


CONFIG_DICT = { 
                    'OneObs2D_ds1_10M': OneObs2D_ds1_10M.config_dict,
            }


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
configname = sys.argv[1]
num        = int(sys.argv[2])
result_dir  = sys.argv[3]

print(f"configname={configname}")
config_dict = CONFIG_DICT[configname]

config = runner.dict2namespace(config_dict)

OneObs = dataset.get_dataset(config)
num_epochs = config.train.num_epochs

save_dir   = REPO_ROOT / config.eval.eval_dir / result_dir / config.config_name
os.makedirs(str(save_dir), exist_ok=True)

gtruth, u_max, u_min, v_max, v_min, x, y ,t = cond_gen.get_data(config, datatype= "Train") #Test
#gtruth_test, u_max, u_min, v_max, v_min, x, y ,t = cond_gen.get_data(config, datatype= "Test") #Test


#Load the generated samples
data_pred_path = config.dataset.pred_data_file
hf =  h5py.File(data_pred_path, 'r')

d1 = hf['seed_1'][:]
d2 = hf['seed_2'][:]
d3 = hf['seed_3'][:]
d4 = hf['seed_4'][:]
d5 = hf['seed_5'][:]

u_pred = np.concatenate( [d1[:,0, :,:], d2[:,0,:, :], d3[:,0,:, :], d4[:,0,:, :], d5[:,0,:, :]] ,  axis = 0 )
v_pred = np.concatenate( [d1[:,1, :, :], d2[:,1,:, :], d3[:,1,:, :], d4[:,1,:, :], d5[:,1,:, :]] ,  axis = 0 )

#renormalize
u_pred = np.clip(u_pred, a_min=-1, a_max =1)
v_pred = np.clip(v_pred, a_min=-1, a_max =1)
u_pred = (1  + u_pred)/2
v_pred = (1  + v_pred)/2
u_pred = u_min + u_pred * (u_max-u_min) 
v_pred = v_min + v_pred * (v_max-v_min)

assert gtruth[:,0].shape[1:] == u_pred.shape[1:]
print(f"Shape: ground truth-{gtruth[:,0].shape}, prediction-{u_pred.shape}")

pred   = utils.combine_fields(u=u_pred, v=v_pred)

# Statistical evaluation of the Model
stats_eval = stats_eval.StatisticalEvaluation(gtruth=gtruth, pred=pred, x_axis=x, y_axis=y, z_axis=None, time=t, 
                    input_data_type = "2D",  data="line-x", config= config)

print(f"Saving pdf at {save_dir}")

pdf = PdfPages(str(save_dir / f'stats_eval-{configname}_ep{num_epochs}_num-{num}-final.pdf'))
pdf = stats_eval.main(num = num, locations = [1, 2, 3, 4], y = 0.5, pdf = pdf)  #locations = locations along x, where the PSD needs to be calculated
pdf.close()
