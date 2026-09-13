import cmocean
import matplotlib.pyplot as plt


plot_dict = dict(
    u_reference_value = 0.7009188319757983 , #np.sqrt(u_pred_max*np.max(u_train))
    figure= dict(
            figsize=(10,5.5),
            dpi=600,
            tight_layout=True,
            obs_pos_x= -0.125,
            obs_pos_y = 0,
            obs_width = 0.25,
            obs_height= 1,
            levels=100,
            re_levels=5,
            jpdf_level=15,
            color_pallete = [
                                "#1f78b4",  # Slate Blue
                                "#e31a1c",  # Coral Red
                                "#33a02c",  # Olive Green
                                "#ffb000",  # Goldenrod
                                "#6a3d9a",  # Charcoal Gray
                                "#1b9e77",  # Teal Blue
                                "#d95f02",  # Burnt Orange
                                "#7570b3",  # Dusty Purple
                                "#006400",  # Forest Green
                                "#e7298a",  # Deep Rose
                                ]
    ),
    
    axes=dict(
            x_label=r'$x/h$',        # X-axis label
            y_label=r'$y/h$',        # Y-axis label
            time_label = r'$t$',      #time
            mse_label = r'$\varepsilon$',
            mse_comp_label = [r'$\varepsilon_{u^{\prime}}$', r'$\varepsilon_{v^{\prime}}$'],
            fluc_label=[r'$u^{\prime}$' , r'$v^{\prime}$' , r'$w^{\prime}$' ],
            mean_label=[r'$\overline{u^{\prime}}$' , r'$\overline{v^{\prime}}$' , r'$\overline{w^{\prime}}$' ],
            re_norm_stresses   =[r"$\overline{u^{\prime}u^{\prime}}$", r"$\overline{v^{\prime}v^{\prime}}$", r"$\overline{u^{\prime}v^{\prime}}$"],  #This has been changed to ease the multiple re_stress label
            re_norm_stresses_p =[r"$\overline{{u^{\prime}_p}{u^{\prime}_p}}$",r"$\overline{{v^{\prime}_p}{v^{\prime}_p}}$", r"$\overline{{u^{\prime}_p}{v^{\prime}_p}}$"],
            re_sh_stresses     =[r"$\overline{u^{\prime}v^{\prime}}$", r"$\overline{u^{\prime}w^{\prime}}$",  r"$\overline{v^{\prime}w^{\prime}}$" ],
            re_sh_stresses_p   =[r"$\overline{{u^{\prime}_p}{v^{\prime}_p}}$",r"$\overline{{u^{\prime}_p}{w^{\prime}_p}}$", r"$\overline{{v^{\prime}_p}{w^{\prime}_p}}$"],
            x_lim=(-1,5),
            y_lim=(0,2),
            fontsize=55,
            vis_compare_fontsize=40,
            re_plane_fontsize = 40,
            re_profiles_fontsize = 40,
            pdfs_fontsize = 40,
            x_ticks=[ -1, 0, 1, 2, 3, 4 ],
            y_ticks=[ 0, 1, 1.8],
            ticksize=50,

    ),
    plot=dict(
        snap_cmap=cmocean.cm.balance, #cmocean.cm.balance,     #'viridis',
        re_stress_cmap=plt.get_cmap("coolwarm"),   #'rainbow',
        jpdf_cmap='RdBu',

    ),

    legend=dict(
            comp_labels=['streamwise', 'wall-normal', 'spanwise'],
            levels=100,

    ),
    )
    

def basic_plt_setup():

        import matplotlib.pyplot as plt
        size = 20 # 40        
        plt.rc("font", family='serif')
        plt.rc("text", usetex='true')
        plt.rc("font", size=size) 
        plt.rc("axes", labelsize=size, linewidth=2)
        plt.rc("legend", fontsize=size, handletextpad=0.1)
        plt.rc("xtick", labelsize=size)
        plt.rc("ytick", labelsize=size)

        return