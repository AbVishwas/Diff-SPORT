import torch
import torchvision
import argparse
import random
import os
import sys
import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, PILToTensor, Normalize, Resize
from .dataset import get_dataset
from .lib_diffusion import make_schedule, forward_diffusion, reverse_diffusion, DDPM_R
from collections import namedtuple
from PIL import Image
from .guided_diffusion.unet import UNetModel
from .guided_diffusion.script_util import create_model
import torch.nn as nn

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def check_conv_layers_precision(model):
    """
    Checks the precision (data type) of all convolutional layers in the model.
    Example usage
    check_conv_layers_precision(model)
    """
    for name, layer in model.named_modules():  # Iterates over all layers
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(layer, 'weight') and layer.weight is not None:
                print(f"Layer {name} precision: {layer.weight.dtype}")
            else:
                print(f"Layer {name} has no weights")



def check_model_precision(model):
    """
    Check and summarize the precision of the model's parameters.
    Example usage:
    for layer_name, stats in precision_summary.items():
       print(f"Layer: {layer_name}, Max Value: {stats['max_value']}, Unique Values: {stats['unique_values']}, Data Type: {stats['dtype']}")

    """
    precision_summary = {}
    for name, param in model.named_parameters():
        max_val = param.data.abs().max()
        unique_values = len(torch.unique(param.data))
        dtype = param.data.dtype
        precision_summary[name] = {
            "max_value": max_val.item(),
            "unique_values": unique_values,
            "dtype": str(dtype)
        }
    return precision_summary

def create_model_and_initialize_ddpm(config_dict):
    
    config = dict2namespace(config_dict)
    #print(f"{config.config_name} Config selected")
    #print(f"{['#']*20}\n\n")
    #print(f"Config Dict = {config_dict}")
    #print(f"{['#']*20}\n\n")

    #model = create_model(**config_dict['model'])
    # 1) create_model does some pre-processing of channel_mult and of attention_resolutions to match at which img resolutions to apply attention at
    #    but it assumes square images and image_size is a scalar  #3D case has different resolutions
    
    
    # 2) I have previously used UNetModel (for Lorenz) directly but this means I have to assume attention_resolutions 
    #    denote 2^depth beforehand where depth is at what level we need attention mechanism
    
    # Fix for 1 and 2 ==> use UNetModel class directly with channel_mult and attention_res args (2^depth) in the config

    # 3) Also for 3D inputs, downsampling is done on last 2 dims, so fix which one has to be the first dimension.. 
    #    or change in Downsample/Upsample class
    
    # Fix for 3 ==> Changed this to downsample/upsample all dims
    
    model = UNetModel(**config_dict["model"])

    precision_summary = check_model_precision(model)
    random_key = random.choice(list(precision_summary.keys()))
    current_data_type = precision_summary[random_key]['dtype']

    if config_dict['model']['use_fp16']:
            model.convert_to_fp16()

    if config.train.MulG_train:
        numgpus = torch.cuda.device_count() 
        print(f"Creating DataParallel model")
        model = torch.nn.DataParallel(model, device_ids=list(range(numgpus)))
    else:
        numgpus = 1

    schedule = make_schedule(scheme=config.diffusion.beta_schedule, rvar='beta', T=config.diffusion.num_diffusion_timesteps, 
                    start_beta=config.diffusion.beta_start, end_beta=config.diffusion.beta_end)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters in Unet: {total_params/1000000} Million', flush=True)

    # DDPM wrapper
    ddpm_model = DDPM_R(schedule=schedule,model=model,weightedloss=config.diffusion.weightedloss,cuda=True)

    return config, ddpm_model, schedule

def load_dataset(config): #config is a namespace here and not a dict

    dataset = get_dataset(config)
    print(f'Dataset has size {len(dataset)}')    

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
        drop_last=False,
        pin_memory=True #Faster transfer to cuda memory via pinning
    )

    return data_loader

def remove_module_prefix(state_dict):
    """
    When we use nn.DataParallel all the state_dicts come with prefix module.
    This is the function to remove that prefix and load model without parallel for inference.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def add_module_prefix(state_dict):
    """
    This function adds the prefix 'module.' to all keys in the state dictionary
    if they don't already have it. This is useful when preparing a model for 
    parallel processing using nn.DataParallel.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            new_key = 'module.' + k  # add 'module.' prefix
        else:
            new_key = k  # keep the original key
        new_state_dict[new_key] = v  # add the key-value pair to the new dict
    
    return new_state_dict


def load_model(config, ddpm_model):
    
    ckpt = torch.load(config.load_model)
    print(f"Loaded Model: {config.load_model}")
    if config.train.MulG_train:
        new_state_dict=add_module_prefix(ckpt['model_state_dict'])
    else:
        new_state_dict = remove_module_prefix(ckpt['model_state_dict'])
    
    ddpm_model.model.load_state_dict(new_state_dict)
    return ddpm_model

def train(config, ddpm_model, data_loader):
    #num_epochs, batch_size, savefreq, prefix
    
    optimizer = torch.optim.Adam(ddpm_model.model.parameters(),lr=config.train.lr)
    num_epochs = config.train.num_epochs
    savefreq = config.train.savefreq
    ckpt_dir = config.train.ckpt_dir
    prefix = config.train.prefix
    
    print(f"Training started for {config.config_name} with image size {config.model.image_size}")
    store_loss = [] ; epochs = [];
    for epoch in range(1,num_epochs+1):
        
        LOSS = 0
        ITER = 0

        pbar = tqdm.tqdm(data_loader)
        for data in pbar:
            data_x = data
            optimizer.zero_grad()
            loss = ddpm_model.run_step(data_x)
            loss.backward()
            optimizer.step()
            
            LOSS += loss.item() * data_x.shape[0]
            ITER += data_x.shape[0]
            
            pbar.set_description(f"LOSS = {round(LOSS/ITER,6)}")

            
        
        print(f"epoch = {epoch}, loss = {LOSS/ITER}") 
        store_loss.append((LOSS/ITER)) ;  epochs.append(epoch) 

        if epoch % savefreq == 0:
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            if not os.path.exists(os.path.join(ckpt_dir,prefix)):
                os.mkdir(os.path.join(ckpt_dir,prefix))
                
            D = {'model_state_dict': ddpm_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch }
            torch.save(D, os.path.join(ckpt_dir, prefix, "ckpt_"+str(epoch)+'.ckpt') ) 


    store_loss = np.array(store_loss).reshape(-1, 1)
    epochs = np.array(epochs).reshape(-1, 1)
    store_loss = np.concatenate((epochs, store_loss), axis = 1)
    print(f"loss is stored, shape:{store_loss.shape}")

    # Save losses under repo-root/train_utils/losses to be CWD-independent
    repo_root = Path(__file__).resolve().parents[1]
    loss_dir = repo_root / 'train_utils' / 'losses'
    loss_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(loss_dir / f'{config.config_name}_den_loss.npz'), loss=store_loss)


def resume(config, ddpm_model, data_loader):
    
    ckpt = torch.load(config.resume.ckpt)

    if config.train.MulG_train:
        new_state_dict=add_module_prefix(ckpt['model_state_dict'])

    else:
        new_state_dict=remove_module_prefix(ckpt['model_state_dict'])

    ddpm_model.model.load_state_dict(new_state_dict)

    optimizer = torch.optim.Adam(ddpm_model.model.parameters(),lr=0.) #intentionally put lr=0., for possible bugs 
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    start_epoch = ckpt['epoch']+1
    num_epochs = config.resume.num_epochs
    batch_size = config.resume.batch_size
    savefreq = config.resume.savefreq
    ckpt_dir = config.resume.ckpt_dir
    prefix = config.resume.prefix

    print(f"Training resumed for {config.dataset.name} with image size {config.model.image_size}, start_epoch = {start_epoch}")

    for epoch in range(start_epoch,num_epochs+1):
        
        LOSS = 0
        ITER = 0
        
        pbar = tqdm.tqdm(data_loader)
        for data in pbar:
            
            data_x = data
            optimizer.zero_grad()
            loss = ddpm_model.run_step(data)
            loss.backward()
            optimizer.step()
            
            LOSS += loss.item() * data_x.shape[0]
            ITER += data_x.shape[0]
            pbar.set_description(f"LOSS = {round(LOSS/ITER,6)}")
        
        print(f"epoch = {epoch}, loss = {LOSS/ITER}")
        if epoch % savefreq == 0:
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            if not os.path.exists(os.path.join(ckpt_dir,prefix)):
                os.mkdir(os.path.join(ckpt_dir,prefix))
                
            D = {'model_state_dict': ddpm_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch }
            torch.save(D, os.path.join(ckpt_dir, prefix, "ckpt_"+str(epoch)+'.ckpt') )

def generate_samples_legacy(config, ddpm_model, onlymean=False): # adapted for arbitrary n-Dimensional data
    
    schedule = make_schedule(scheme=config.diffusion.beta_schedule, rvar='beta', T=config.diffusion.num_diffusion_timesteps, 
                        start_beta=config.diffusion.beta_start, end_beta=config.diffusion.beta_end)

    num_samples = config.sampling.batch_size
    c = config.model.in_channels
    imgsize = config.model.image_size
    z = np.random.normal(loc=0, scale=1.0, size=(num_samples,c,*imgsize))
    z = np.asarray(z, dtype=np.float32)
    #----------------------------------------------------------------

    T = config.diffusion.num_diffusion_timesteps
    x = z
    for t in range(0,T):
        x = reverse_diffusion(ddpm_model=ddpm_model, x=x, t_curr=T-t, cuda=True, schedule=schedule, onlymean=onlymean)
    pred_samples = x
    assert pred_samples.shape == z.shape
    #----------------------------------------------------------------
    return pred_samples

def generate_samples(config, ddpm_model, sampler=None, num_samples=None, sample_batch_size=None, T_sub=None, eta=None, onlymean=False):
    
    if num_samples is None:
        num_samples = config.sampling.num_samples
    if sample_batch_size is None:
        sample_batch_size = config.sampling.batch_size
    c = config.model.in_channels
    imgsize = config.model.image_size
    sample_shape = (c,*imgsize)
    if eta is None:
        eta = config.sampling.eta
    if T_sub is None:
        T_sub = config.sampling.T_sub
    
    return ddpm_model.generate_samples(num_samples=num_samples, sample_batch_size=sample_batch_size, sample_shape=sample_shape, sampler=sampler, T_sub=T_sub, eta=eta, onlymean=onlymean)
