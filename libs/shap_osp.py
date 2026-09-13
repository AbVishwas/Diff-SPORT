import numpy as np
import torch
import torch.nn.functional as F
from libs.lib_svd import Inpainting_custom, Inpainting_custom_mask_batches
from libs import cond_gen, utils, inst_eval
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_pdf import PdfPages

class shap_osp:
    def __init__(self, schedule, model, gtruth_tensor = None, 
                       sigma_y=None, weightedloss=True, cuda=True,
                        T_sub=20, onlymean=False, adaptive=False, 
                        y_c=None, class_cond=None,
                        zero_mask= None, full_mask= None,
                        seg_tensor=None, u_max=None, u_min=None, 
                        v_max=None, v_min=None, x_axis =None, y_axis=None, time=None, unique_segments=None,
                        zzero=None, zshap=None, train_mean=None, config=None, T_sub_mapgd=None, gditer_mapgd = None):


        # Initialize the additional parameters specific to shap_osp
        self.model = model
        self.gtruth_tensor = gtruth_tensor
        self.sigma_y = sigma_y
        self.T_sub = T_sub
        self.onlymean = onlymean
        self.adaptive = adaptive
        self.y_c = y_c
        self.class_cond = class_cond
        self.cuda = cuda
        self.zero_mask = zero_mask
        self.full_mask = full_mask
        self.seg_tensor = seg_tensor
        self.u_max = u_max
        self.u_min = u_min
        self.v_max = v_max
        self.v_min = v_min
        self.x_axis=x_axis
        self.y_axis=y_axis
        self.time=time
        self.unique_segments=unique_segments
        self.zzero =zzero
        self.zshap =zshap
        self.train_mean=train_mean
        self.config=config
        self.T_sub_mapgd=T_sub_mapgd
        self.gditer_mapgd=gditer_mapgd

    def shap_wrapper_mapgd(self, mask_tensor):   

        # Use the instance attributes
        H = Inpainting_custom(mask_tensor=mask_tensor, device="cuda")
        obs = H.H(self.gtruth_tensor)

        #print(f"Mask tensor shape: {mask_tensor.shape} and ground truth: {self.gtruth_tensor_torch.shape}")
        assert mask_tensor.shape == self.gtruth_tensor.shape[1:]

        x_output_mapgd = self.model.reverse_diffusion_gd(self.gtruth_tensor.shape,  obs, self.sigma_y, H=H, T_ddrm=self.T_sub_mapgd, cuda=self.cuda, onlymean=self.onlymean, 
                             karras=False, adaptive=self.adaptive, y_c=self.y_c, class_cond=self.class_cond, gditer=self.gditer_mapgd)
        
        x_output_mapgd = cond_gen.renormalize(x_output_mapgd.cpu().numpy(), self.u_max, self.u_min, self.v_max, self.v_min)
        
        loss = F.mse_loss(torch.tensor(x_output_mapgd).to("cuda"), self.gtruth_tensor, reduction='none')
        
        print(f"Shapes:obs:{obs.shape}, x_output_mapgd:{x_output_mapgd.shape}, self.gtruth_tensor:{self.gtruth_tensor.shape}, loss:{loss.shape} ")
        
        loss = loss.mean(dim=(2, 3))

        return loss.cpu().numpy(), x_output_mapgd


    def shap_wrapper_pgdm(self, mask_tensor, H, obs):   

        #print(f"Mask tensor shape: {mask_tensor.shape} and ground truth: {self.gtruth_tensor_torch.shape}")
        assert mask_tensor.shape == self.gtruth_tensor.shape[1:] 

        x_output_pgdm = self.model.reverse_diffusion_pgdm(self.gtruth_tensor.shape, obs, self.sigma_y, 
                                                    H=H, T_ddrm=self.T_ddrm, cuda=self.cuda, 
                                                    onlymean=self.onlymean, adaptive=self.adaptive, 
                                                    y_c=self.y_c, class_cond=self.class_cond)
        
        x_output_pgdm = cond_gen.renormalize(x_output_pgdm.cpu().numpy(), self.u_max, self.u_min, self.v_max, self.v_min)
        loss = F.mse_loss(torch.tensor(x_output_pgdm).to("cuda"), self.gtruth_tensor, reduction='none')

        loss = loss.mean(dim=(2, 3))
        return loss.cpu().numpy(), x_output_pgdm

    def mask_dom(self, zs):

            """
            Function for making the domain
            """
            # If no background is defined the mean value of the field is taken
            background = self.zero_mask.clone()
            mask_out = self.full_mask.clone()
            assert background.shape == mask_out.shape

            # Replace the values of the field in which the feature is deleted
            for jj in range(zs.shape[0]): 
                if zs[jj] == 0:
                    mask_out[self.seg_tensor == self.unique_segments[jj]] = background[self.seg_tensor == self.unique_segments[jj]]
                    
            return mask_out


    def mask_dom2(self, zii):
            """
            Function for making the domain
            """

            # If no background is defined the mean value of the field is taken
            background = self.zero_mask.clone()
            mask_out = self.full_mask.clone()
            #updated_gtruth = self.gtruth_tensor.clone()
            assert background.shape == mask_out.shape

            # Replace the values of the field in which the feature is deleted
            for jj in range(zii.shape[0]): 

                if zii[jj] == self.zzero[0,jj]:
                    indices = torch.nonzero(self.seg_tensor == self.unique_segments[jj], as_tuple=True)
                    #values_u = self.train_mean[:,  0, indices[1], indices[2]]
                    #updated_gtruth[:, 0, indices[1], indices[2] ] =  torch.mean(values_u, dim=1)
                    #values_v = self.train_mean[:,  1, indices[1], indices[2]]
                    #updated_gtruth[:, 1, indices[1], indices[2] ] =  torch.mean(values_v, dim=1)

                    mask_out[self.seg_tensor == self.unique_segments[jj]] = background[self.seg_tensor == self.unique_segments[jj]]

            #H = Inpainting_custom(mask_tensor=self.full_mask, device="cuda") ##If we use full mask here, the model will not predict for the near ground region.
            #updated_gtruth = updated_gtruth * self.full_mask.unsqueeze(0)  #This is not required as similar operation will occur in the generation (Just as safe option)

            #updated_gtruth = torch.from_numpy(cond_gen.normalize(updated_gtruth.cpu().numpy(), self.u_max, self.u_min, self.v_max, self.v_min)).to("cuda")
            #obs = H.H(updated_gtruth)                 

            return mask_out   #, H, obs, updated_gtruth


    def model_function(self, zs):
        ii = 0
        lm = zs.shape[0]
        mse = np.zeros((lm,2))

        start_time = utils.get_current_time()
        print(f"Starting to run SHAP configurations @ {start_time}")

        for zii in zs:

            config_start_time = utils.get_current_time()
            #mask_out, H, obs, updated_gtruth = self.mask_dom(zii)
            mask_out = self.mask_dom2(zii)
            # mse[ii,0] =  self.shap_wrapper(H, obs)
            mse[ii,:], x_output_pgdm =  self.shap_wrapper_mapgd(mask_out) 
   
            if int(ii) in [int(x * (1)) for x in range(10)]:
                cond_gen.plot_mask(mask_tensor=mask_out.cpu().numpy(), mask=30, x_axis=self.x_axis, y_axis=self.y_axis, mask_type=f"SHAP{ii}-{lm}" , out_pdf=True)

                ############
                # Remove this
                gtruth = self.gtruth_tensor.cpu().numpy()
                output = x_output_pgdm

                # Calculate the error
                _, error = utils.calculate_mse(gtruth, output,  self.config.eval.plot_dict.u_reference_value)

                # Create the figure and axes
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Normalize the color scale across all images
                vmin = gtruth.min()
                vmax = gtruth.max()

                vmin_error = 0
                vmax_error = 0.1

                # Plot Ground Truth (axis=1 slices)
                img1 = axes[0, 0].imshow(gtruth[0, 0, :, ::-1].T, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, 0].set_title("Ground Truth Slice 1")
                axes[0, 0].axis('off')

                img2 = axes[1, 0].imshow(gtruth[0, 1, :, ::-1].T, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1, 0].set_title("Ground Truth Slice 2")
                axes[1, 0].axis('off')

                # Plot Output (axis=1 slices)
                img3 = axes[0, 1].imshow(output[0, 0, :, ::-1].T, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, 1].set_title("Output Slice 1")
                axes[0, 1].axis('off')

                img4 = axes[1, 1].imshow(output[0, 1, :, ::-1].T, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1, 1].set_title("Output Slice 2")
                axes[1, 1].axis('off')

                # Plot Difference/Error (axis=1 slices)
                img5 = axes[0, 2].imshow(error[0, 0, :, ::-1].T, cmap='viridis', vmin=vmin_error, vmax=vmax_error)
                axes[0, 2].set_title("Difference Slice 1")
                axes[0, 2].axis('off')

                img6 = axes[1, 2].imshow(error[0, 1, :, ::-1].T, cmap='viridis', vmin=vmin_error, vmax=vmax_error)
                axes[1, 2].set_title("Difference Slice 2")
                axes[1, 2].axis('off')

                # Add a single colorbar for all images
                cbar = fig.colorbar(img1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
                cbar.set_label('Velocity')

                cbar1 = fig.colorbar(img5, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
                cbar1.set_label('Error')

                # Adjust layout for better spacing
                #plt.tight_layout()
                plt.savefig(f"./temp/check-output-{int(ii)}-{lm}.pdf")
                plt.close()

                ############

            config_end_time = utils.get_current_time()
            print(f"Calculated SHAP values for config#{ii}/{lm} in {utils.get_time_difference(config_start_time, config_end_time)}, mse_u @ {mse[ii,0]},mse_v @ {mse[ii,1]}, num_pixels: {torch.count_nonzero(mask_out[0])}",flush=True)
            
            ii += 1

        end_time = utils.get_current_time()
        print(f"MSE calculations for all configurations finished in {utils.get_time_difference(start_time, end_time)} !!!",flush=True)

        #mse_mag = np.sqrt(mse[:, 0]**2 + mse[:, 1]**2)

        return mse




###############################################################################################
#Functions for batch implementations, gives > 60x speedup
##################################################################################################
    def shap_wrapper_mapgd_mask_batches(self, H, obs_batch, data_norm_batch, seed=0):   

        x_output_mapgd_norm = self.model.reverse_diffusion_gd_mask_batches(data_norm_batch.shape,  obs_batch, self.sigma_y, H=H, T_ddrm=self.T_sub_mapgd, cuda=self.cuda, onlymean=self.onlymean, 
                             karras=False, adaptive=self.adaptive, y_c=self.y_c, class_cond=self.class_cond, gditer=self.gditer_mapgd, seed=seed)
        
        loss = F.mse_loss(x_output_mapgd_norm.clone().detach().to("cuda"), data_norm_batch, reduction='none')
        loss = loss.mean(dim=(2, 3))

        return loss.cpu().numpy(), x_output_mapgd_norm

    def mask_dom_mask_batches(self, zs):
            """
            Function to generate domain masks.

            Parameters:
                zs (torch.Tensor): Input tensor of shape (batch_size, features).

            Returns:
                torch.Tensor: Stacked masks of shape (batch_size, *mask_shape).
            """

            # If no background is defined the mean value of the field is taken
            background = self.zero_mask.clone()
            masks = []
            
            time1 = utils.get_current_time()

            # Validate input dimensions
            if zs.shape[1] != self.zzero.shape[1]:
                raise ValueError("Mismatch between zs and zzero feature dimensions.")

            # Replace the values of the field in which the feature is deleted
            for ii in range(zs.shape[0]):
                mask_out = self.full_mask.clone()
                for jj, unique_segment in enumerate(self.unique_segments):
                    if zs[ii,jj] == self.zzero[0,jj]:
                        mask_out[self.seg_tensor == unique_segment] = background[self.seg_tensor == unique_segment]

                masks.append(mask_out)
            time2 = utils.get_current_time()
            print(f"{len(masks)} masks are prepared in {utils.get_time_difference(time1, time2)} !!")
            return torch.stack(masks, dim=0) 

    def plot_batch_results(self, masks_batch, data_norm_batch, x_output_mapgd_batch_norm, batch_idx, num_batches):
        data_batch = cond_gen.renormalize(data_norm_batch, self.u_max, self.u_min, self.v_max, self.v_min).cpu().numpy()
        x_output_batch = cond_gen.renormalize(x_output_mapgd_batch_norm, self.u_max, self.u_min, self.v_max, self.v_min).cpu().numpy()
        
        ground_truth_min, ground_truth_max = np.min(data_batch), np.max(data_batch)
        error_min, error_max = 0, 0.01
        
        for k in range(data_batch.shape[0]):
            cond_gen.plot_mask(masks_batch[k].cpu().numpy(), mask=30, x_axis=self.x_axis, y_axis=self.y_axis,
                            mask_type=f"SHAP{k}-{num_batches}", out_pdf=True)
            
            fig, axes = plt.subplots(2, 3, figsize=(24, 12))
            for j in range(2):
                utils.plot_image(axes[j, 0], data_batch[k, j], "Ground Truth", ground_truth_min, ground_truth_max)
                utils.plot_image(axes[j, 1], x_output_batch[k, j], "Reconstructed", ground_truth_min, ground_truth_max)
                _, error = utils.calculate_mse(data_batch[k, j], x_output_batch[k, j], self.config.eval.plot_dict.u_reference_value)
                utils.plot_image(axes[j, 2], error, "Error", error_min, error_max)
            
            utils.add_colorbars(fig, ground_truth_min, ground_truth_max, error_min, error_max)
            plt.savefig(f"./temp/check-output-batch{k}-{num_batches}.pdf")
            plt.close()



    def model_function_mask_batches(self, zs, gen_seeds=[0,1,2,3] ,check_first_batch=False):

        #ii = 0
        lm = zs.shape[0]
        batch_size = max(self.config.sampling.batch_size, 1)
        num_batches = max(lm // batch_size, 1)

        #mse = np.zeros((lm,2))

        config_start_time = utils.get_current_time()
        print(f"Starting SHAP configurations @ {config_start_time}, lm: {lm}", flush=True)

        masks = self.mask_dom_mask_batches(zs) 
        print(f"Shape of masks: {masks.shape}", flush=True)
        
        data_norm = cond_gen.normalize(self.gtruth_tensor, self.u_max, self.u_min, self.v_max, self.v_min)

        if data_norm.shape[0] != masks.shape[0]:
            if data_norm.shape[0] > masks.shape[0]:
                # Repeat `masks` to match the size of `data_norm` in the first dimension
                repeat_factor = (data_norm.shape[0] + masks.shape[0] - 1) // masks.shape[0]  # Ceiling division
                masks = masks.repeat(repeat_factor, 1, 1, 1)[:data_norm.shape[0], ...]
            elif data_norm.shape[0] < masks.shape[0]:
                # Repeat `data_norm` to match the size of `masks` in the first dimension
                repeat_factor = (masks.shape[0] + data_norm.shape[0] - 1) // data_norm.shape[0]  # Ceiling division
                data_norm = data_norm.repeat(repeat_factor, 1, 1, 1)[:masks.shape[0], ...]

        assert data_norm.shape == masks.shape

        if lm == 1 :
            batch_size=1
            num_batches=1

        #mse = []
        all_best_errors = []
        for i in range(num_batches):

            masks_batch = masks[i*batch_size:(i+1)*batch_size].cuda()

            #print(f"masks_batch:{masks_batch.shape}")
            H = Inpainting_custom_mask_batches(mask_tensor=masks_batch, device=masks_batch.device) 

            data_norm_batch  = data_norm[i*batch_size:(i+1)*batch_size].cuda()
            obs_batch = H.H(data_norm_batch) 
            instance_errors = []

            for gen_seed in gen_seeds:
                mse_batch, x_output_mapgd_batch_norm =self.shap_wrapper_mapgd_mask_batches(H, obs_batch, data_norm_batch, seed=gen_seed )
                mse_mag_batch = np.linalg.norm(mse_batch, axis=1)
                instance_errors.append(mse_mag_batch.reshape(-1,1))

            instance_errors = np.concatenate(instance_errors, axis=1)
            best_indices = np.argmin(instance_errors, axis=1)
            best_errors = instance_errors[np.arange(len(instance_errors)), best_indices].reshape(-1, 1)
            good_fields = np.count_nonzero(best_errors <= 0.005)

            config_end_time = utils.get_current_time()
            num_pixels = torch.count_nonzero(masks_batch[:, 0, :, :], dim=(1, 2)).unsqueeze(1) 
            
            print(f"best_indices: {best_indices} @{i} batches in {utils.get_time_difference(config_start_time, config_end_time)} % good fields={good_fields}/{batch_size}", flush=True)
            all_best_errors.append(best_errors)

        all_best_errors = np.concatenate(all_best_errors, axis=0)    

        return all_best_errors


    def model_function_mask_batchesv2(self, zs, gen_seeds=[0] ,check_first_batch=True):
        
        lm = zs.shape[0]
        batch_size = max(self.config.sampling.batch_size, 1)  # Ensure batch size is at least 1
        num_batches = max(lm // batch_size, 1)

        config_start_time = utils.get_current_time()
        print(f"Starting SHAP configurations @ {config_start_time}, lm: {lm}", flush=True)
 
        masks     = self.mask_dom_mask_batches(zs)
        data_norm = cond_gen.normalize(self.gtruth_tensor, self.u_max, self.u_min, self.v_max, self.v_min)

        # Get shapes
        masks_size, data_norm_size = masks.shape[0], data_norm.shape[0]

        if data_norm_size != masks_size:
            max_size = max(masks_size, data_norm_size)

            # Pad along batch dimension (dim=0)
            if data_norm_size < max_size:
                pad_size = max_size - data_norm_size
                padding_shape = (0, 0) * (data_norm.dim() - 1)  # No padding for other dims
                data_norm = F.pad(data_norm, (*padding_shape, 0, pad_size), mode="constant", value=0)

            if masks_size < max_size:
                pad_size = max_size - masks_size
                padding_shape = (0, 0) * (masks.dim() - 1)  # No padding for other dims
                masks = F.pad(masks, (*padding_shape, 0, pad_size), mode="constant", value=0)

        assert data_norm.shape == masks.shape
        
        all_best_errors = []
        for i in range(num_batches):
            masks_batch = masks[i * batch_size : (i + 1) * batch_size]
            data_norm_batch = data_norm[i * batch_size : (i + 1) * batch_size]
            
            H = Inpainting_custom_mask_batches(mask_tensor=masks_batch, device=masks_batch.device)
            obs_batch = H.H(data_norm_batch)

            instance_errors = []
            
            for gen_seed in gen_seeds:
                print(f"0: {utils.get_current_time()}")
                mse_batch, x_output_mapgd_batch_norm = self.shap_wrapper_mapgd_mask_batches(H, obs_batch, data_norm_batch, seed=gen_seed)
                print(f"1: {utils.get_current_time()}")
                mse_mag_batch = np.linalg.norm(mse_batch, axis=1)
                
                instance_errors.append(mse_mag_batch.reshape(-1,1))

            instance_errors = np.concatenate(instance_errors, axis=1)
            best_indices = np.argmin(instance_errors, axis=1)
            best_errors = instance_errors[np.arange(len(instance_errors)), best_indices].reshape(-1, 1)

            good_fields = np.count_nonzero(best_errors <= 0.005)

            if i == 1 and check_first_batch:
                self.plot_batch_results(masks_batch, data_norm_batch, x_output_mapgd_batch_norm, i, num_batches)
            
            config_end_time = utils.get_current_time()
            
            all_best_errors.append(best_errors)
        
        all_best_errors = np.concatenate(all_best_errors, axis=0)
        print(f"all_best_errors:{all_best_errors.shape}, best_errors:{best_errors.shape}, mse_batch:{mse_batch.shape}, mse_mag_batch:{mse_mag_batch.shape}")
        
        return all_best_errors



def get_shap_features(seg_tensor = None,  snap=None):
    """
    Process segments to compute mean values for unique indices in `seg_tensor`.

    Parameters:
    unique_values : array-like
        Unique values in the segmentation tensor.
    seg_tensor : np.ndarray
        Tensor containing segmentation data.
    snap : np.ndarray
        Mean snapshot for "missing" features (background) and a single snapshot for "present" features (zshap) (Must be from velocity magnitude)
    Returns:
    z : np.ndarray
        "missing" features (background) or "present" features (zshap)
    """
    if seg_tensor is None or seg_tensor.numel() == 0 or len(seg_tensor.shape) < 2:
        raise ValueError("seg_tensor must be a non-empty tensor with at least 2 dimensions")

    if snap is None or not isinstance(snap, np.ndarray) or len(snap.shape) != 2:
        raise ValueError("a single snapshot, u_mag")

    if isinstance(seg_tensor, torch.Tensor):
        seg_tensor = seg_tensor.cpu().numpy()

    unique_values = np.unique(seg_tensor)
    nmax2 = len(unique_values) - 1 # Neglecting the zero (non-mask region) in unique values 
    z = np.zeros((1, nmax2)) 

    # Iterate through unique values
    for i, seg_index in enumerate(unique_values):

        # Find indices where seg_tensor matches seg_index
        indices = np.argwhere(seg_tensor[0] == seg_index) 

        # Access values from snap using the indices
        values = snap[tuple(indices.T)]

        if seg_index >= 1.0:
            z[0, i - 1] = np.mean(values)

    return z
