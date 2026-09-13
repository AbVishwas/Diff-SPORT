import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import time
from typing import Optional
import os
from matplotlib import cm

from libs.utils import get_data_for_stats, add_obstacle_patch, plot_subplot, plot_subplot_contourf
from configs.plot_config import basic_plt_setup
basic_plt_setup()

### Class to evaluate statistics for given ground truth and predictions.
### Gives a pdf file containing a visual comparison of single snap, reynolds stresses (along a line and plane),
### Probability density functions for velocity fields and power spectral density plots.

class StatisticalEvaluation:

    def __init__(self, gtruth=None, pred=None, x_axis=None, y_axis=None, z_axis=None, time=None, 
                      input_data_type = "2D",  data=None, config= None, Train_data=True, Test_data=False):
        """
        Initialize the StatisticalEvaluation class with the provided parameters and evaluate the DDPM from statistical sense.

        Parameters:
        gtruth (ndarray): Ground truth data.
        pred (ndarray): Predicted data.
        x_axis (ndarray): X-axis values.
        y_axis (ndarray): Y-axis values.
        z_axis (ndarray): Z-axis values.
        time (ndarray): Time values.
        input_data_type (str): Type of input data ('2D' or '3D').
        data (ndarray): Additional data.
        config (object): Configuration object.
        """

        assert gtruth is not None and pred is not None, "Ground truth and prediction must not be None"
        assert gtruth.any() and pred.any(), "Ground truth and prediction must not be empty"

        self.gtruth  = gtruth
        self.pred    = pred

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.time   = time

        self.input_data_type = input_data_type
        self.data            = data
        self.config          = config
        self.plot_config       = config.eval.plot_dict
        self.Train = Train_data
        self.Test  = Test_data

        # Obstacle dimensions & location
        self.pos_x, self.pos_y  = self.plot_config.figure.obs_pos_x, self.plot_config.figure.obs_pos_y
        self.width, self.height = self.plot_config.figure.obs_width, self.plot_config.figure.obs_height  # width, height of the obstacle

        self.min, self.max = np.min(self.gtruth), np.max(self.gtruth)

        self.eval_dir = config.eval.eval_dir + '/results/'

        self.x_label = self.plot_config.axes.x_label
        self.y_label = self.plot_config.axes.y_label
        self.fontsize = self.plot_config.axes.fontsize
     
        #plt.rcParams['font.family'] = 'serif'
        #plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

        if input_data_type == "2D":
            assert self.gtruth.shape[1] == self.pred.shape[1] == 2 , "Expected 2 velocity components in 2D data"
            assert len(self.gtruth.shape) == len(self.pred.shape) == 4, "Expected 2D data to have 4 dimensions (time, nc ,u, v)"

        elif input_data_type == "3D":
            assert self.gtruth.shape[1] == self.pred.shape[1] == 3 , "Expected 3 velocity components in 3D data"
            assert len(self.gtruth.shape) == len(self.pred.shape) == 5, "Expected 3D data to have 5 dimensions (time, nc ,u, v, w)"

        if not os.path.exists(self.eval_dir):
            os.mkdir(self.eval_dir)
        if not os.path.exists(os.path.join(self.eval_dir,self.config.config_name)):
            os.mkdir(os.path.join(self.eval_dir,self.config.config_name))

    def plot_vis_compare(self, num=None, pdf =None):
        """
        Plot visual comparison of ground truth and predicted data.

        Parameters:
        num (int): Index for the time step to plot.
        pdf (PdfPages): PDF object to save the plots.
        """

        labels = self.plot_config.legend.comp_labels
        levels = self.plot_config.legend.levels


        if self.input_data_type == "2D":
            fig, axs = plt.subplots(2, 3, figsize=(3*self.plot_config.figure.figsize[0], 2*self.plot_config.figure.figsize[1]))

        elif self.input_data_type == "3D":
            fig, axs = plt.subplots(2, 3, figsize=(3*self.plot_config.figure.figsize[0], 2*self.plot_config.figure.figsize[1]))

        else:
            raise ValueError("Unsupported input data type")

        for i in range(self.gtruth.shape[1]):
            # Plot Ground Truth

            extent = [self.x_axis.min(), self.x_axis.max(), self.y_axis.min(), self.y_axis.max()]
            
            img1 = plot_subplot(axs[i,0], self.gtruth[num, i, :, :],  extent=extent, vmin=self.min, vmax= self.max, fontsize=self.plot_config.axes.vis_compare_fontsize, colormap=self.plot_config.plot.snap_cmap)
            img2 = plot_subplot(axs[i,1], self.pred[num, i, :, :], extent=extent,    vmin=self.min, vmax= self.max, fontsize=self.plot_config.axes.vis_compare_fontsize, colormap=self.plot_config.plot.snap_cmap)
            img3 = plot_subplot(axs[i,2], self.pred[num+1, i, :, :], extent=extent,  vmin=self.min, vmax= self.max, fontsize=self.plot_config.axes.vis_compare_fontsize, colormap=self.plot_config.plot.snap_cmap)
            
        if self.Train:
            plt.savefig(self.eval_dir  + self.config.config_name +'/' f'visual_comparison_{self.config.config_name}_num-{num}.png', dpi=self.plot_config.figure.dpi)
            
        if self.Test:
            plt.savefig(self.eval_dir  + self.config.config_name +'/' f'visual_comparison_{self.config.config_name}_num-{num}-test.pdf', dpi=self.plot_config.figure.dpi)

        if pdf:
            fig.set_dpi(self.plot_config.figure.dpi)  
            pdf.savefig(fig)

        plt.close(fig)

        return pdf


    def reynolds_stress(self, inp=None, x = None, y = None, z =None, data = None):
        """
        Compute Reynolds stresses for the given input data.

        Parameters:
        inp (ndarray): Input data.
        x (ndarray): X-axis values.
        y (ndarray): Y-axis values.
        z (ndarray): Z-axis values.
        data (str): Data type ('line-x', 'plane', etc.).


        Returns:
        tuple: Reynolds stress components.
        """
        
        if self.input_data_type == "3D":
            u, v, w = inp[:, 0], inp[:, 1],  inp[:, 2]
            assert len(u.shape) == 4, "Expected 3D data to have 4 dimensions (time, x, y, z)"
        
        elif self.input_data_type == "2D":
            u, v, w = inp[:, 0], inp[:, 1], None
            assert len(u.shape) == 3, "Expected 2D data to have 3 dimensions (time, x, y)"

        #print(self.config.dataset.ds_ratio)
        # Extract the relevant data for statistics
        u_pt = get_data_for_stats(u, x= x, y= y,  z= z, input_data_type=self.input_data_type, data=data, ds_ratio =self.config.dataset.ds_ratio, mean_over_time=False)
        v_pt = get_data_for_stats(v, x= x, y= y,  z= z, input_data_type=self.input_data_type, data=data , ds_ratio =self.config.dataset.ds_ratio, mean_over_time=False)

        # Compute the Reynolds stresses, Dont know why I am not using mean_overtime=True
        rs_uu = np.mean(u_pt * u_pt, axis=0)
        rs_vv = np.mean(v_pt * v_pt, axis=0)
        rs_uv = np.mean(np.abs(u_pt * v_pt), axis=0)

        if w == None:
            rs_ww = None
            rs_vw = None
            rs_uw = None
        else:
            w_pt = get_data_for_stats(w, x=x, y=y, z=z, input_data_type=self.input_data_type, data=data, mean_over_time=False)
            rs_ww = np.mean(w_pt * w_pt, axis=0)
            rs_vw = np.mean(v_pt * w_pt, axis=0)
            rs_uw = np.mean(u_pt * w_pt, axis=0)

        return rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw

    def get_mean_vel_profiles(self, inp=None, x=None, y=None, z=None, data=None):

        if self.input_data_type == "3D":
            u, v, w = inp[:, 0], inp[:, 1],  inp[:, 2]
            assert len(u.shape) == 4, "Expected 3D data to have 4 dimensions (time, x, y, z)"
        
        elif self.input_data_type == "2D":
            u, v, w = inp[:, 0], inp[:, 1], None
            assert len(u.shape) == 3, "Expected 2D data to have 3 dimensions (time, x, y)"        

        # Extract the relevant data for statistics
        u_pt = get_data_for_stats(u, x= x, y= y,  z= z, input_data_type=self.input_data_type, data=data, ds_ratio =self.config.dataset.ds_ratio, mean_over_time=False)
        v_pt = get_data_for_stats(v, x= x, y= y,  z= z, input_data_type=self.input_data_type, data=data , ds_ratio =self.config.dataset.ds_ratio, mean_over_time=False)

        print(f"shape: u_pt: {u_pt.shape},v_pt: {v_pt.shape}")

        u_pt = np.mean(u_pt, axis =0)
        v_pt = np.mean(v_pt, axis =0)

        print(f"shape: u_pt: {u_pt.shape},v_pt: {v_pt.shape}")

        return u_pt, v_pt


    def plot_mean_vel_profiles_multiple_locations(self, locations=None, x=None, y=None, z=None, data='line-y', pdf=None):
        # Validate locations
        if locations is None:
            raise ValueError("Please provide valid locations for plotting.")        

        # Initialize lists to accumulate Reynolds stresses
        u_m, v_m= [], []
        u_m_pred, v_m_pred = [], []

        # Compute and accumulate Reynolds stresses for each location
        for loc in locations:
            means = self.get_mean_vel_profiles(inp=self.gtruth, x=loc, y=y, z=z, data=data)
            means_pred = self.get_mean_vel_profiles(inp=self.pred, x=loc, y=y, z=z, data=data)

            u_m.append(means[0])
            v_m.append(means[1])

            u_m_pred.append(means_pred[0])
            v_m_pred.append(means_pred[1])

        # Convert accumulated lists to arrays for easier processing
        u_m, v_m = map(np.array, [u_m, v_m])
        u_m_pred, v_m_pred = map(np.array, [u_m_pred, v_m_pred])

        # Select appropriate plot axis
        plot_axis = self.x_axis if x is None and z is None else self.y_axis
        plot_x_label = self.x_label if x is None and z is None else self.y_label

        # Colormap for multiple locations
        colors = cm.viridis(np.linspace(0, 1, u_m.shape[0]))

        # Reynolds stresses and their predictions
        mean_vel_profiles = [u_m, v_m]
        mean_vel_profiles_p = [u_m_pred, v_m_pred]
        mean_labels = self.plot_config.axes.mean_label
        mean_titles = mean_labels
        # Plotting loop
        for j in range(len(mean_vel_profiles)):   
            fig, ax = plt.subplots(figsize=(2.5*self.plot_config.figure.figsize[1], 2.5*self.plot_config.figure.figsize[1]))
            for i, color in enumerate(colors):
                ax.plot(plot_axis, mean_vel_profiles[j][i], label=rf"$x/h = {locations[i]}$", linestyle='-', color=color)
                ax.plot(plot_axis, mean_vel_profiles_p[j][i], label=rf"$x/h = {locations[i]}$", linestyle='None', marker='o', color=color)

            # Set labels, title, and legend
            ax.set_xlabel(plot_x_label)
            ax.set_ylabel(mean_labels[j]) 
            ax.set_title(mean_titles[j])  
            ax.legend(loc="best") 

            if self.plot_config.figure.tight_layout:
                fig.tight_layout()
            
        if self.Train:
            plt.savefig(self.eval_dir + self.config.config_name + f'/mean_vel_profiles_{self.config.config_name}_comp_{j}.pdf', dpi=self.plot_config.figure.dpi)

        if self.Test:
            plt.savefig(self.eval_dir + self.config.config_name + f'/mean_vel_profiles_{self.config.config_name}_comp_{j}-test.pdf', dpi=self.plot_config.figure.dpi)

        if pdf:
            fig.set_dpi(self.plot_config.figure.dpi)  
            pdf.savefig(fig)            
            
        plt.close(fig)

        return pdf


    def plot_reynolds_stresses_multiple_locations(self, locations=None, x=None, y=None, z=None, data='line-y', pdf=None):
        # Validate locations
        if locations is None:
            raise ValueError("Please provide valid locations for plotting.")        

        # Initialize lists to accumulate Reynolds stresses
        rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = [], [], [], [], [], []
        rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = [], [], [], [], [], []

        # Compute and accumulate Reynolds stresses for each location
        for loc in locations:
            stresses = self.reynolds_stress(inp=self.gtruth, x=loc, y=y, z=z, data=data)
            stresses_pred = self.reynolds_stress(inp=self.pred, x=loc, y=y, z=z, data=data)

            rs_uu.append(stresses[0])
            rs_vv.append(stresses[1])
            rs_ww.append(stresses[2])
            rs_uv.append(stresses[3])
            rs_vw.append(stresses[4])
            rs_uw.append(stresses[5])

            rs_uu_pred.append(stresses_pred[0])
            rs_vv_pred.append(stresses_pred[1])
            rs_ww_pred.append(stresses_pred[2])
            rs_uv_pred.append(stresses_pred[3])
            rs_vw_pred.append(stresses_pred[4])
            rs_uw_pred.append(stresses_pred[5])

        # Convert accumulated lists to arrays for easier processing
        rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = map(np.array, [rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw])
        rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = map(np.array, [rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred])

        # Select appropriate plot axis
        plot_axis = self.x_axis if x is None and z is None else self.y_axis
        plot_x_label = self.x_label if x is None and z is None else self.y_label

        # Colormap for multiple locations
        colors =self.plot_config.figure.color_pallete    #cm.viridis(np.linspace(0, 1, rs_uu.shape[0]))

        # Reynolds stresses and their predictions
        reynolds_stresses = [rs_uu, rs_vv, rs_uv]
        reynolds_stresses_p = [rs_uu_pred, rs_vv_pred, rs_uv_pred]
        stress_labels = self.plot_config.axes.re_norm_stresses
        stress_titles = stress_labels
        # Plotting loop
        for j in range(len(reynolds_stresses)):   
            fig, ax = plt.subplots(figsize=(1*self.plot_config.figure.figsize[0], 1*self.plot_config.figure.figsize[0]))
            for i, color in enumerate(colors[:rs_uu.shape[0]]):
                ax.plot(plot_axis, reynolds_stresses[j][i], label=rf"$x/h = {locations[i]}$", linestyle='-', color=color)
                ax.plot(plot_axis, reynolds_stresses_p[j][i], label=rf"$x/h = {locations[i]}$", linestyle='--', color=color)

            # Set labels, title, and legend
            ax.set_xlabel(plot_x_label, fontsize=self.plot_config.axes.re_profiles_fontsize)
            ax.set_ylabel(stress_labels[j], fontsize=self.plot_config.axes.re_profiles_fontsize) 
            #ax.set_title(stress_titles[j])  

            ax.tick_params(axis='both', labelsize=self.plot_config.axes.re_profiles_fontsize-10)
            ax.legend(fontsize=self.plot_config.axes.re_profiles_fontsize-10, ncol=3) 

            if self.plot_config.figure.tight_layout:
                fig.tight_layout()

            # Save the plot
            save_path = self.eval_dir + self.config.config_name + f'/Reynolds_stresses_{self.config.config_name}_comp_{j}.png'
            #fig_legend.savefig("legend_only.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

            if pdf:
                pdf.savefig(fig)
            plt.savefig(save_path, dpi=self.plot_config.figure.dpi)
            plt.close(fig)

        return pdf


    def plot_reynolds_stresses(self, x=None, y=None, z=None, data='line-x', pdf =None):
        """
        Plot Reynolds stresses for ground truth and predicted data.

        Parameters:
        x (ndarray): X-axis values.
        y (ndarray): Y-axis values.
        z (ndarray): Z-axis values.
        data (str): Data type ('line-x', 'plane', etc.).
        pdf (PdfPages): PDF object to save the plots.
        """
        rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = self.reynolds_stress(inp= self.gtruth, x=x, y=y, z=z, data=data)
        rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = self.reynolds_stress(inp=self.pred, x=x, y=y, z=z , data=data)
        
        # Check if rs_ww and rs_ww_pred are None
        is_3d = rs_ww is not None and rs_ww_pred is not None

        # Plotting
        fig, ax = plt.subplots(figsize=(2*self.plot_config.figure.figsize[1], 2*self.plot_config.figure.figsize[1]))

        ax.plot(self.x_axis, rs_uu, label=self.plot_config.axes.re_norm_stresses[0], linestyle='-', color='b')
        ax.plot(self.x_axis, rs_uu_pred, label=self.plot_config.axes.re_norm_stresses_p[0], linestyle='None', marker='o', color='b')
        
        ax.plot(self.x_axis, rs_vv, label=self.plot_config.axes.re_norm_stresses[1], linestyle='-', color='g')
        ax.plot(self.x_axis, rs_vv_pred, label=self.plot_config.axes.re_norm_stresses_p[1], linestyle='None', marker='^', color='g')
        
        ax.plot(self.x_axis, rs_uv, label=self.plot_config.axes.re_sh_stresses[0], linestyle='-', color='r')
        ax.plot(self.x_axis, rs_uv_pred, label=self.plot_config.axes.re_sh_stresses_p[0], linestyle='None', marker='v', color='r')

        if is_3d:
            ax.plot(self.x_axis, rs_ww, label=self.plot_config.axes.re_norm_stresses[2], linestyle='-', color='c')
            ax.plot(self.x_axis, rs_ww_pred, label=self.plot_config.axes.re_norm_stresses_p[2], linestyle='None', marker='s', color='c')
            
            ax.plot(self.x_axis, rs_uw, label=self.plot_config.axes.re_sh_stresses[1], linestyle='-', color='m')
            ax.plot(self.x_axis, rs_uw_pred, label=self.plot_config.axes.re_sh_stresses_p[1], linestyle='None', marker='d', color='m')
            
            ax.plot(self.x_axis, rs_vw, label=self.plot_config.axes.re_sh_stresses[2], linestyle='-', color='y')
            ax.plot(self.x_axis, rs_vw_pred, label=self.plot_config.axes.re_sh_stresses_p[2], linestyle='None', marker='*', color='y')

        ax.set_xlabel(self.x_label)  #, fontsize = self.plot_config.axes.fontsize)
        ax.set_ylabel(r"$\overline{{u_i}^{\prime}{u_j}^{\prime}}$") #, fontsize=self.plot_config.axes.fontsize)
        ax.legend()    # prop={'size': self.plot_config.axes.fontsize})
        ax.set_title('Reynolds Stress Components')#, fontsize = self.plot_config.axes.fontsize)
        
        if self.plot_config.figure.tight_layout:
            fig.tight_layout()
        
        plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'Reynolds_stresses1_{self.config.config_name}.pdf', dpi=self.plot_config.figure.dpi)
        
        if pdf != None:
            pdf.savefig(fig)
            
        plt.close(fig)

        return pdf


    def _plot_contour(self, ax, Y_grid, X_grid, data, data_pred, title, levels, cmap, fontsize, cbar_orientation="vertical"):
        """
        Helper method to plot a single contour plot with ground truth and predicted data.
        #TODO: Move this to utils
        Parameters:
        ax (Axes): The matplotlib Axes object to plot on.
        Y_grid (ndarray): Y-axis grid values.
        X_grid (ndarray): X-axis grid values.
        data (ndarray): Ground truth data.
        data_pred (ndarray): Predicted data.
        title (str): Title of the plot.
        levels (int): Number of contour levels.
        cmap (str): Colormap.
        fontsize (int): Font size for the title.
        cbar_orientation (str): Orientation of the colorbar ('vertical' or 'horizontal'). Default is 'vertical'.
        """

        # Create filled contour plot for ground truth
        contour = ax.contourf(Y_grid, X_grid, data, levels=levels, cmap=cmap)
        
        # Create contour plot for predicted data in black
        contour_pred = ax.contour(Y_grid, X_grid, data_pred, levels=levels, colors='black', linestyles='dashed')

        # Add obstacle patch to the plot (custom method)
        add_obstacle_patch(ax)

        # Set title and axis labels
        #ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(self.plot_config.axes.x_label, fontsize=fontsize)
        ax.set_ylabel(self.plot_config.axes.y_label, fontsize=fontsize)

        # Set ticks and their sizes
        ax.set_xticks(self.plot_config.axes.x_ticks)
        ax.set_yticks(self.plot_config.axes.y_ticks)
        ax.tick_params(axis='both', labelsize=fontsize-10)

        # Get the figure object from the axes
        fig = ax.get_figure()

        # Add a colorbar with the correct orientation and tick size
        cbar = fig.colorbar(contour, ax=ax, orientation=cbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize-10)



    def plot_reynolds_stress_planes(self, x=None, y=None, z=None, data='plane', pdf=None):
        """
        Plot Reynolds stress planes for ground truth and predicted data.

        Parameters:
        x (ndarray): X-axis values.
        y (ndarray): Y-axis values.
        z (ndarray): Z-axis values.
        data (str): Data type ('plane', etc.).
        levels (int): Number of contour levels.
        pdf (PdfPages): PDF object to save the plots.

        Returns:
        pdf (PdfPages): PDF object with the saved plots.
        """
        
        rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = self.reynolds_stress(self.gtruth, x=x, y=y, z=z, data=data)
        rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = self.reynolds_stress(self.pred, x=x, y=y, z=z, data=data)

        # Check if rs_ww and rs_ww_pred are None
        is_3d = rs_ww is not None and rs_ww_pred is not None

        # Create mesh grid for plotting
        X_grid, Y_grid = np.meshgrid(self.y_axis, self.x_axis)

        cmap = self.plot_config.plot.re_stress_cmap
        fontsize = self.plot_config.axes.re_plane_fontsize
        levels=self.plot_config.figure.re_levels
        
        if not is_3d:
            fig, axs = plt.subplots(1, 3, figsize=(3*self.plot_config.figure.figsize[0], 1*self.plot_config.figure.figsize[1]))

            # Plot uu
            self._plot_contour(axs[0], Y_grid, X_grid, rs_uu, rs_uu_pred, self.plot_config.axes.re_norm_stresses[0], levels, cmap, fontsize)

            # Plot vv
            self._plot_contour(axs[1], Y_grid, X_grid, rs_vv, rs_vv_pred, self.plot_config.axes.re_norm_stresses[1], levels, cmap, fontsize)

            # Plot uv
            self._plot_contour(axs[2], Y_grid, X_grid, rs_uv, rs_uv_pred, self.plot_config.axes.re_sh_stresses[0], levels, cmap, fontsize)

        else:

            fig, axs = plt.subplots(3, 2, figsize=(5*self.plot_config.figure.figsize[0], 2*self.plot_config.figure.figsize[1]))
            # Plot uu
            self._plot_contour(axs[0, 0], Y_grid, X_grid, rs_uu, rs_uu_pred, self.plot_config.axes.re_norm_stresses[0], levels, cmap, fontsize)

            # Plot vv
            self._plot_contour(axs[0, 1], Y_grid, X_grid, rs_vv, rs_vv_pred, self.plot_config.axes.re_norm_stresses[1], levels, cmap, fontsize)

            # Plot ww
            self._plot_contour(axs[1, 0], Y_grid, X_grid, rs_ww, rs_ww_pred, self.plot_config.axes.re_norm_stresses[2], levels, cmap, fontsize)

            # Plot uv
            self._plot_contour(axs[1, 1], Y_grid, X_grid, rs_uv, rs_uv_pred, self.plot_config.axes.re_sh_stresses[0], levels, cmap, fontsize)

            # Plot uw
            self._plot_contour(axs[2, 0], Y_grid, X_grid, rs_uw, rs_uw_pred, self.plot_config.axes.re_sh_stresses[1], levels, cmap, fontsize)

            # Plot vw
            self._plot_contour(axs[2, 1], Y_grid, X_grid, rs_vw, rs_vw_pred, self.plot_config.axes.re_sh_stresses[2], levels, cmap, fontsize)

        if self.plot_config.figure.tight_layout:
            fig.tight_layout()

        if self.Train:
            plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'Reynolds_stresses2_{self.config.config_name}.png', dpi=self.plot_config.figure.dpi)

        if self.Test:
            plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'Reynolds_stresses2_{self.config.config_name}-test.pdf', dpi=self.plot_config.figure.dpi)

        if pdf:
            fig.set_dpi(self.plot_config.figure.dpi)  
            pdf.savefig(fig)   
            
        plt.close(fig)

        return pdf

    def joint_pdfs(self, inp_comp = None, axis = None):
        """
        Calculate the joint probability density function (PDF) of input components and their corresponding axis values.

        Parameters:
        inp_comp (ndarray): The input components to calculate the PDF for.
        axis (ndarray): The axis values corresponding to the input components.

        Returns:
        xi (ndarray): The meshgrid x-values for contour plotting.
        yi (ndarray): The meshgrid y-values for contour plotting.
        zi_norm (ndarray): The normalized joint PDF values for the meshgrid.
        """
        from scipy.stats import gaussian_kde

        u_copy = inp_comp.reshape(-1)
        x_copy = (np.tile(axis.reshape(1, inp_comp.shape[1]), (inp_comp.shape[0],1))).reshape(-1) 

        # Calculate the point density
        xy = np.vstack([x_copy, u_copy])
        z = gaussian_kde(xy)

        # Create a grid for contour plotting
        #print(u_.min())
        xi, yi = np.linspace(axis.min(), axis.max(), 100), np.linspace(inp_comp.min(), inp_comp.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

        zi_norm = zi/np.max(zi)

        return xi, yi, zi_norm


    def plot_joint_pdfs(self, x=None, y=None, z=None, data='line-x', pdf=None):
        """
        Plot the joint probability density functions (PDFs) for the components of the ground truth and predicted data.

        Parameters:
        x (ndarray): The x-coordinates for data selection.
        y (ndarray): The y-coordinates for data selection.
        z (ndarray): The z-coordinates for data selection.
        data (str): The type of data to process ('line-x', 'line-y', etc.).
        levels (int): The number of contour levels to plot.
        pdf (PdfPages): An optional PdfPages object to save the plots to a PDF file.

        Returns:
        pdf (PdfPages): The PdfPages object if provided, with the plots saved.
        """
        levels=self.plot_config.figure.jpdf_level
        labels =self.plot_config.axes.fluc_label
        fontsize=self.plot_config.axes.pdfs_fontsize
        x_skip = 57 #skip the inflow region in x

        for i in range(self.gtruth.shape[1]):
            
            component      = get_data_for_stats(self.gtruth[:,i,x_skip:, :], x=x, y=y, z=z, input_data_type = self.input_data_type,  data=data, ds_ratio=self.config.dataset.ds_ratio)
            component_pred = get_data_for_stats(self.pred[:,i,x_skip:, :], x=x, y=y, z=z, input_data_type = self.input_data_type,  data=data, ds_ratio=self.config.dataset.ds_ratio)
 
            #print("creating pdfs")
            xi, yi, zi_norm      = self.joint_pdfs(component, axis = self.x_axis[x_skip:]); #print("gtruth done")
            xi, yi, zi_norm_pred = self.joint_pdfs(component_pred, axis = self.x_axis[x_skip:]); #print("pred done")

            # Plotting
            fig, ax = plt.subplots(figsize=(self.plot_config.figure.figsize[0], self.plot_config.figure.figsize[0]))

            # Use a perceptually uniform colormap (e.g., magma) for DNS
            contour = ax.contourf(xi, yi, zi_norm, levels=levels,
                                cmap='coolwarm', alpha=0.9, antialiased=True)

            # Overlay DDPM prediction as dashed contours
            contour_pred = ax.contour(xi, yi, zi_norm_pred, levels=levels,
                                    colors='black', linestyles='dashed', linewidths=2.0)

            #plt.title("Joint PDF of $x$ and $u'$", fontsize=self.fontsize)
            ax.set_xlabel(self.plot_config.axes.x_label, fontsize=fontsize)
            ax.set_ylabel(labels[i], fontsize=fontsize)

            ax.tick_params(axis='both', labelsize=fontsize-10)
            
            cbar = fig.colorbar(contour, ax=ax)
            cbar.ax.tick_params(labelsize=fontsize-10)

            #ax.set_ylim(-0.5, 0.5)
            
            fig.tight_layout()

            if self.Train:
                plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'jpdfs-{i}_{self.config.config_name}@y:{y}.png', dpi=self.plot_config.figure.dpi)

            if self.Test:
                plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'jpdfs-{i}_{self.config.config_name}@y:{y}-test.pdf', dpi=self.plot_config.figure.dpi)

            if pdf:
                fig.set_dpi(self.plot_config.figure.dpi)  
                pdf.savefig(fig) 


        return pdf
    
    def PSD_welch(self, inp_comp = None, nperseg=256):
        """
        Compute the Power Spectral Density (PSD) of an input component using the Welch method.

        Parameters:
        inp_comp (ndarray): The input component for which the PSD is to be computed.
        nperseg (int): Length of each segment for the Welch method (default is 256).

        Returns:
        frequencies (ndarray): Array of sample frequencies.
        psd (ndarray): Power spectral density of the input component.
        """
        from scipy.signal import welch

        inp_comp = inp_comp.flatten()

        fs = self.time[1]-self.time[0]

        frequencies, psd = welch(inp_comp, fs=fs, nperseg=nperseg)

        return frequencies, psd


    def plot_PSD(self, locations=None, y=None, z=None, data='point', nperseg=256, pdf = None):
        """
        Plot the Power Spectral Density (PSD) for ground truth and predicted data using the Welch method.

        Parameters:
        locations (list): List of x locations for point data.
        y (float): y-coordinate for line-x data.
        z (float): z-coordinate for line-x data.
        data (str): Type of data, either 'point' or 'line-x'.
        nperseg (int): Length of each segment for the Welch method (default is 256).
        pdf (PdfPages): PdfPages object to save plots to a PDF file.

        Returns:
        pdf (PdfPages): PdfPages object with saved figures.
        """
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#e377c2', '#17becf']
        # Blue, Green, Red, Orange, Magenta, Cyan
        labels =  ['u', 'v', 'w']
        # Create the fig and axis
        linewidth = 2

        if data == "point":
            assert self.gtruth.shape[1] == self.pred.shape[1]
            for k in range(self.gtruth.shape[1]):

                fig, ax = plt.subplots(figsize=(12, 8))

                for i, x in enumerate(locations):

                    # Get ground truth and predicted data
                    comp_x = get_data_for_stats(self.gtruth[:, k], x=x, y=y, z=z, input_data_type=self.input_data_type, data=data, ds_ratio=self.config.dataset.ds_ratio)
                    comp_x_pred = get_data_for_stats(self.pred[:, k], x=x, y=y, z=z, input_data_type=self.input_data_type, data=data, ds_ratio=self.config.dataset.ds_ratio)

                    # Compute PSD using Welch's method
                    frequency, psd = self.PSD_welch(comp_x, nperseg=nperseg)
                    frequency, psd_p = self.PSD_welch(comp_x_pred, nperseg=nperseg)

                    # Plot the PSD
                    ax.loglog(frequency, psd, label=fr"GT @ $\frac{{x}}{{h}}=$ {x}", linestyle='-', linewidth=linewidth, color=colors[i])
                    ax.loglog(frequency, psd_p, label=fr'pred @ $\frac{{x}}{{h}}=$ {x}', color=colors[i], marker='o', markersize=5, linestyle='none')

                # Add titles and labels
                ax.set_title('Power Spectral Density vs Frequency (Welch Method)', fontsize=self.fontsize)
                ax.set_xlabel('Frequency [Hz]', fontsize=self.fontsize)
                ax.set_ylabel('PSD [V**2/Hz]', fontsize=self.fontsize)

                # Customize grid and legend
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

                # Tight layout for better spacing
                fig.tight_layout()

                if self.Train:
                    plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'PSD-pt-{labels[k]}_{self.config.config_name}.pdf', dpi=self.plot_config.figure.dpi)

                if self.Test:
                    plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'PSD-pt-{labels[k]}_{self.config.config_name}-test.png', dpi=self.plot_config.figure.dpi)

                if pdf:
                    fig.set_dpi(self.plot_config.figure.dpi)  
                    pdf.savefig(fig) 
                
                plt.close(fig)
        
        elif data == "line-x":

            for k in range(self.gtruth.shape[1]):

                fig, ax = plt.subplots(figsize=(12, 8))

                # Get ground truth and predicted data
                comp_x = get_data_for_stats(self.gtruth[:, k], x=None, y=y, z=z, input_data_type=self.input_data_type, data=data, ds_ratio=self.config.dataset.ds_ratio)
                comp_x_pred = get_data_for_stats(self.pred[:, k], x=None, y=y, z=z, input_data_type=self.input_data_type, data=data, ds_ratio=self.config.dataset.ds_ratio)

                # Compute PSD using Welch's method
                frequency, psd = self.PSD_welch(comp_x, nperseg=nperseg)
                frequency, psd_p = self.PSD_welch(comp_x_pred, nperseg=nperseg)

                # Plot the PSD
                ax.loglog(frequency, psd, label=fr"GT @ $\frac{{y}}{{h}}=$ {y}", linestyle='-', linewidth=linewidth, color='k')
                ax.loglog(frequency, psd_p, label=fr'pred @ $\frac{{y}}{{h}}=$ {y}', color='k', marker='o', markersize=5, linestyle='none')

                # Add titles and labels
                ax.set_title('Power Spectral Density vs Frequency (Welch Method)', fontsize=16)
                ax.set_xlabel('Frequency [Hz]', fontsize=14)
                ax.set_ylabel('PSD [V**2/Hz]', fontsize=14)

                # Customize grid and legend
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

                # Tight layout for better spacing
                fig.tight_layout()

                if self.Train:
                    plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'PSD_linex-{labels[k]}_{self.config.config_name}.pdf', dpi=self.plot_config.figure.dpi)

                if self.Test:
                    plt.savefig(self.eval_dir + self.config.config_name +'/'+ f'PSD_linex-{labels[k]}_{self.config.config_name}-test.pdf', dpi=self.plot_config.figure.dpi)

                if pdf:
                    fig.set_dpi(self.plot_config.figure.dpi)  
                    pdf.savefig(fig) 
                
                plt.close(fig)


        return pdf


    def plot_probe_signal(self,  n = 200,  x = None, y = None, z = None):

        labels = ['u', 'v', 'w']
        for i in range(self.gtruth.shape[1]):

            u_pt = get_data_for_stats(self.gtruth[:, i], x=x, y=y, z=z, input_data_type = "2D",  data="point", mean_over_time=False, ds_ratio= config.dataset.ds_ratio)# Check this while integrating the function
            u_pt_p = get_data_for_stats(self.pred[:, i], x=x, y=y, z=z, input_data_type = "2D",  data="point", mean_over_time=False, ds_ratio= config.dataset.ds_ratio)

            print(u_pt.shape); print(u_pt_p.shape)
            random_integers = np.random.randint(0, u_pt.shape[0], n)

            # Plot settings
            plt.figure(figsize=(12, 8))

            plt.plot(u_pt[random_integers], label=rf"${labels[i]}'$ @"  +r"$\frac{x}{h}= 1$ ", linestyle='-', color='k')
            plt.plot(u_pt_p[:n],  label=rf"${labels[i]}_p'$ @"  +r"$\frac{x}{h}= 1$ ", linestyle='-', color='r')

            # Add grid lines
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Add titles and labels
            plt.xlabel('num of snapshots', fontsize=14)
            plt.ylabel(rf"${labels[i]}'$", fontsize=14)

            # Add a legend
            plt.legend(loc='best', fontsize=12)

            # Add limits for x-axis to ensure both series are easily comparable
            plt.xlim([0, n])
            plt.ylim([-.5, 0.5])
            # Add a scientific look with a tighter layout
            plt.tight_layout()

            # Show the plot
            plt.show()


    def main(self, num = None, locations = None, y = None , pdf = None):
        """
        Main function to execute various plotting and analysis routines.

        Parameters:
        num (int): The index or identifier for the visual comparison (optional).
        locations (list): List of x locations for PSD plotting (optional).
        y (float): y-coordinate for line-x data in various plots (optional).
        pdf (PdfPages): PdfPages object to save plots to a PDF file (optional).

        Returns:
        pdf (PdfPages): PdfPages object with saved figures.
        """
        print("Getting a visual glimps of prediction")
        pdf = self.plot_vis_compare(num= num, pdf = pdf)


        print("Computing Reynolds Stresses")
        pdf = self.plot_reynolds_stresses(x = None, y = 0.5, z = None, pdf = pdf )
        pdf = self.plot_reynolds_stresses_multiple_locations(locations=locations, x=1, y=None, z=None, data='line-y', pdf=pdf)
        pdf = self.plot_reynolds_stress_planes(x = None, y = None, z = None, data='plane', pdf = pdf)

        print("Getting pdfs for flow fields")
        pdf = self.plot_joint_pdfs(x = None, y = 0.05, z = None, data = 'line-x', pdf = pdf)
        pdf = self.plot_joint_pdfs(x = None, y = 0.5, z = None, data = 'line-x', pdf = pdf)
        pdf = self.plot_joint_pdfs(x = None, y = 1.0, z = None, data = 'line-x', pdf = pdf)

        print("Computing Power Spectral Density")
        pdf = self.plot_PSD(locations = locations, y= 0.5, z=None, data='point', nperseg=256, pdf = pdf)
        pdf = self.plot_PSD(locations = None, y= 0.5, z=None, data='line-x', nperseg=256, pdf = pdf)

        return pdf


if __name__ == "__main__":
    import sys
    sys.path.append('../')

    from configs.OneObs2D_ds1_10M import config_dict
    from libs import runner
    from libs.utils import get_data_for_stats

    from matplotlib.backends.backend_pdf import PdfPages


    print(f"Class to evaluate statistics for given ground truth and predictions. Outputs a pdf file with relevant statistics")
    print(f"Starting dummy implementation!! ")
    # Dummy data and configuration
    gtruth = np.random.rand(5, 2, 60, 20)  # Example ground truth data 
    pred = np.random.rand(5, 2, 60, 20)    # Example predicted data 
    x_axis = np.linspace(-1, 5, 60)
    y_axis = np.linspace(0, 2, 20)
    time = np.linspace(0, 10, 5)  # Example time array

    config = runner.dict2namespace(config_dict)

    # Instantiate the class
    evaluator = StatisticalEvaluation(gtruth=gtruth, pred=pred, x_axis=x_axis, y_axis=y_axis, z_axis=None, time=time, input_data_type = "2D",  data="line-x", config= config)
    
    # Setup PDF output
    pdf = PdfPages(f'./test_stats_eval.pdf')
    pdf = evaluator.main(num = 0, locations = [0.5], y = 0.5, pdf = pdf)  #locations = locations along x, where the PSD needs to be calculated
    pdf.close()