from configs.plot_config import plot_dict
import re
from pathlib import Path

# Resolve repository root (diffSPORT directory)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Build absolute paths from repo root for portability across CWDs
train_data_file = str(REPO_ROOT / 'data' / 'train_data_file.h5')
test_data_file  = str(REPO_ROOT / 'data' / 'test_data_file.h5')

pred_data_file  = str(REPO_ROOT / 'inference_utils' / 'unconditional_generation' / 'generated_samples' / 'generated_samples-25k-OneObs2D_ds1_10M_ep1000.h5')
ckpt_dir        = str(REPO_ROOT / 'ckpts') + '/'
eval_dir        = str(REPO_ROOT / 'evaluate_utils') + '/'
cond_gen_dir    = str(REPO_ROOT / 'inference_utils' / 'conditional_generation' / 'generated_samples') + '/'
osp_dir         = str(REPO_ROOT / 'osp_utils') + '/'

prefix = "OneObs2D_ds1_10M"

inf_epochs=1000

resume_ckpt_path = ckpt_dir + prefix +  f'/ckpt_{inf_epochs}.ckpt'
inference_ckpt_path = ckpt_dir + prefix + f'/ckpt_{inf_epochs}.ckpt'  

config_dict = dict(
                
                config_name = prefix,
                dataset = dict(
                        name = "OneObs2D",
                        data_file = train_data_file,
                        test_data_file = test_data_file,
                        pred_data_file= pred_data_file, 
                        filetype = "hdf5",
                        transform = None,
                        ds_ratio = 1,
                        normalize = True,
                        ),

                model = dict(
                            image_size=(288, 96),
                            in_channels=2,
                            model_channels=64,
                            out_channels=2,
                            num_res_blocks=2,
                            attention_resolutions=[4,8], # attentions at lev 1 and 3, indicates 2^depth
                            dropout=0,
                            channel_mult=(1, 2, 2, 2),
                            conv_resample=True,
                            dims=2,
                            num_classes=None,
                            use_checkpoint=True,
                            use_fp16=False,
                            use_fp16_for_data=False,
                            num_heads=1,  #attention heads
                            num_head_channels=-1,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=False,
                            resblock_updown=False,
                            use_new_attention_order=False,
                        ),

                diffusion = dict(
                        beta_schedule = 'linear',
                        beta_start = 0.0001,
                        beta_end = 0.02,
                        num_diffusion_timesteps = 1000,
                        weightedloss = False,
                        ),

                train = dict(
                        batch_size = 32, #32, 64, 128,256
                        shuffle = True,
                        num_workers = 8,
                        lr = 5e-4,       
                        num_epochs = 1000,  # global_epochs
                        savefreq = 50,
                        prefix = prefix, 
                        ckpt_dir = ckpt_dir,
                        MulG_train = True,
                        MulGMulN_train =False, 
                        ),

                sampling = dict(
                        batch_size = 100, 
                        sampler= 'ddim',
                        num_samples= 1200 ,
                        T_sub_mapgd = 20,
                        gditer_mapgd = 50, 
                        T_sub = 1000,   #can vary from 200 to 1000, larger the better performance but more time consuming
                        eta = 1.0 ,
                        onlymean = False,
                        sample_batch_size = 100,
                        inf_epochs=inf_epochs,
                        
                        ),

                resume = dict(
                              ckpt = resume_ckpt_path, #full path of ckpt
                              batch_size = 256,
                              num_epochs = 1000,  # global_epochs, training resumes from ckpt_epoch until global_epochs
                              savefreq = 50,  
                              prefix = prefix,
                              ckpt_dir = ckpt_dir,

                        ),

                eval = dict(
                              eval_dir = eval_dir,
                              cond_gen_dir=cond_gen_dir,
                              osp_dir = osp_dir,
                              plot_dict = plot_dict,

                        ),

                load_model = inference_ckpt_path,

                )

if __name__ == "__main__":
        print(config_dict)
