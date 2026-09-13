import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, gaussian_kde
from libs.utils import calculate_mse, plot_inst_comp_and_error, error_evolution, plot_two_comps
from configs.plot_config import basic_plt_setup

basic_plt_setup()

class InstantaneousEvaluation:
    def __init__(self, gtruth, pred1=None, pred2=None, x_axis=None, y_axis=None, 
                 z_axis=None, time=None, input_data_type="2D", config=None):
        if gtruth is None or not gtruth.any():
            raise ValueError("Ground truth data (gtruth) must not be empty.")
        if (pred1 is None or not pred1.any()) and (pred2 is None or not pred2.any()):
            raise ValueError("At least one prediction (pred1 or pred2) must be provided.")
        if x_axis is None or y_axis is None:
            raise ValueError("Both x_axis and y_axis must be provided.")

        self.gtruth = gtruth
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred = pred1 if pred1 is not None else pred2

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.time = time
        self.extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]

        self.input_data_type = input_data_type
        self.config = config

        self.plot_config = config.eval.plot_dict
        self.u_ref_value = config.eval.plot_dict.u_reference_value

        self.min, self.max = np.min(gtruth), np.max(gtruth)
        self.eval_dir = os.path.join(config.eval.eval_dir, 'results', config.config_name)
        os.makedirs(self.eval_dir, exist_ok=True)

        # Perform evaluations
        if pred1 is not None and pred2 is not None:
            self._evaluate_two_predictions()
        elif pred1 is None or pred2 is None:
            self._evaluate_single_prediction()

    def _evaluate_two_predictions(self):
        """Evaluates and compares two predictions. TODO:This is not tested"""
        if self.input_data_type == "2D":
            assert self.gtruth.shape[1] == self.pred1.shape[1] == self.pred2.shape[1] == 2, \
                "Expected 2 velocity components in 2D data."
            assert len(self.gtruth.shape) == len(self.pred1.shape) == len(self.pred2.shape) == 4, \
                "Expected 2D data to have 4 dimensions (time, nc, u, v)."

            self.time_avg_error1, self.inst_error1 = calculate_mse(self.gtruth, self.pred1, self.u_ref_value)
            self.time_avg_error2, self.inst_error2 = calculate_mse(self.gtruth, self.pred2, self.u_ref_value)
            self.time_space_avg_error1 = np.round(np.mean(self.time_avg_error1), 4)
            self.time_space_avg_error2 = np.round(np.mean(self.time_avg_error2), 4)

            # Store a default view of the first prediction so downstream helpers keep working.
            self.time_avg_error = self.time_avg_error1
            self.inst_error = self.inst_error1
            self.time_space_avg_error = self.time_space_avg_error1

            # Keep individual MSE traces for optional comparison plots later on.
            self.mse1 = np.mean(self.inst_error1, axis=(2, 3))
            self.mse2 = np.mean(self.inst_error2, axis=(2, 3))
            self.mse = self.mse1
        elif self.input_data_type == "3D":
            raise NotImplementedError("Evaluation for 3D data is not implemented yet.")

    def _evaluate_single_prediction(self):
        """Evaluates a single prediction."""
        self.time_avg_error, self.inst_error = calculate_mse(self.gtruth, self.pred, self.u_ref_value)
        self.time_space_avg_error = np.round(np.mean(self.time_avg_error), 4)
        self.mse = np.mean(self.inst_error, axis=(2,3))
        print(f"The error for entire test dataset: {self.time_space_avg_error}")

    def plot_comparisons_and_errors(self, random_indices=None, mask=None, comp="streamwise", 
                                    pdf=None, wake_region=False, ds_ratio=1, 
                                    plot_bound_min=None, plot_bound_max=None):
        """Plots comparisons and errors."""
        plot_bound_min = plot_bound_min or self.min
        plot_bound_max = plot_bound_max or self.max

        fig = plot_inst_comp_and_error(
            random_indices=random_indices, test_data=self.gtruth, pred_data=self.pred,
            error=self.inst_error, mask=mask, comp=comp, x_axis=self.x_axis, y_axis=self.y_axis,
            colormap=self.plot_config.plot.snap_cmap, wake_region=wake_region, ds_ratio=ds_ratio,
            vmin=plot_bound_min, vmax=plot_bound_max
        )

        filename = f"Inst_comp_{self.config.config_name}_{comp}_&_errors.png"
        plt.savefig(os.path.join(self.eval_dir, filename), dpi=self.plot_config.figure.dpi)

        if pdf:
            pdf.savefig(fig)
        plt.close(fig)

        return pdf

    def plot_error_with_time(self, pdf=None):

        fig = error_evolution(errors=self.mse, x_axis= self.time, 
                            x_label = self.plot_config.axes.time_label, 
                            y_label=self.plot_config.axes.mse_label, 
                            legend_labels=self.plot_config.axes.mse_comp_label)

        filename = f"Error_evolution_with_time.png"
        plt.savefig(os.path.join(self.eval_dir, filename), dpi=self.plot_config.figure.dpi)

        if pdf:
            pdf.savefig(fig)

        plt.close(fig)

        return pdf

    def plot_time_avg_error(self, pdf=None):

        fig = plot_two_comps(data = self.time_avg_error, x_axis=self.x_axis, y_axis=self.y_axis,
        x_label=self.plot_config.axes.x_label, y_label=self.plot_config.axes.y_label, vmin=0, vmax=0.003, colormap="cividis", obs_color='lightgrey')

        filename = f"time_avg_errors.png"
        plt.savefig(os.path.join(self.eval_dir, filename), dpi=self.plot_config.figure.dpi)

        if pdf:
            pdf.savefig(fig)

        plt.close(fig)

        return pdf


    def main(self, random_indices=None, mask=None, ds_ratio=1, pdf=None):
        """Main entry point for evaluation and plotting."""
        
        pdf = self.plot_comparisons_and_errors(random_indices=random_indices, mask=mask, comp="streamwise", 
                                                wake_region=False, ds_ratio=ds_ratio, pdf=pdf )

        pdf = self.plot_comparisons_and_errors(random_indices=random_indices, mask=mask, comp="wall-normal", 
                                                wake_region=False, ds_ratio=ds_ratio, pdf=pdf )

        #pdf = self.plot_comparisons_and_errors(random_indices=random_indices, mask=mask, comp="streamwise", 
        #                                        wake_region=True, ds_ratio=ds_ratio, pdf=pdf )

        #pdf = self.plot_comparisons_and_errors(random_indices=random_indices, mask=mask, comp="wall-normal", 
        #                                        wake_region=True, ds_ratio=ds_ratio, pdf=pdf )

        pdf = self.plot_error_pdf_comparison(pdf=pdf)

        return pdf

    def plot_error_pdf_comparison(self, pdf=None, dpi=600):
        errors_dict = { 'MAPGA': self.mse , 'PGDM': self.mse2} if self.pred2 is not None else {'MAPGA': self.mse}
        return plot_pdf_comparison_to_pdf(errors_dict, pdf, dpi=dpi)


def plot_pdf_comparison(errors_dict, save_dir=None, out_pdf=False, out_show=False, out_png=False, dpi=600):
    """
    Plots PDF comparisons between different error datasets.

    Parameters:
    -----------
    errors_dict : dict
        Keys are labels, values are 2D arrays of errors with shape (N, D).
    save_dir : str or None
        Directory to save output files.
    out_pdf : bool
        If True, save the figure as a PDF.
    out_show : bool
        If True, display the plot interactively.
    out_png : bool
        If True, save the figure as a PNG.
    dpi : int
        Resolution for saved figures.
    """

    if not isinstance(errors_dict, dict) or not errors_dict:
        raise ValueError("errors_dict must be a non-empty dict of label -> 2D ndarray")

    first_key = next(iter(errors_dict))
    arr = errors_dict[first_key]
    if arr.ndim != 2:
        raise ValueError("Each errors_dict value must be a 2D array (N, D)")
    n_dims = arr.shape[1]

    fig, axs = plt.subplots(1, n_dims, figsize=(8 * n_dims, 8))
    if n_dims == 1:
        axs = [axs]

    mse_comp_label = [r'$\\varepsilon_{u^{\\prime}}$', r'$\\varepsilon_{v^{\\prime}}$']

    for dim in range(n_dims):
        ax = axs[dim]
        x_min, x_max = np.inf, -np.inf

        # First pass: determine a shared x range using Gaussian fits
        for label, data in errors_dict.items():
            data_dim = np.asarray(data)[:, dim]
            mu, std = norm.fit(data_dim)
            x_vals = np.linspace(mu - 4 * std, mu + 4 * std, 200)
            x_min = min(x_min, x_vals[0])
            x_max = max(x_max, x_vals[-1])

        # Second pass: plot smooth PDFs via KDE
        for label, data in errors_dict.items():
            data_dim = np.asarray(data)[:, dim]
            kde = gaussian_kde(data_dim, bw_method=0.4)
            x_vals = np.linspace(np.min(data_dim), np.max(data_dim), 200)
            pdf_vals = kde.evaluate(x_vals)
            ax.plot(x_vals, pdf_vals, label=label, linewidth=2)
            ax.fill_between(x_vals, pdf_vals, alpha=0.2)
            ax.tick_params(axis='both', labelsize=30)

            # Domain-specific limits (as in the notebook)
            ax.set_xlim(0, 0.035)
            ax.set_ylim(0, 400)

        ax.set_xlabel(mse_comp_label[dim] if dim < len(mse_comp_label) else f"dim {dim}", fontsize=50)
        ax.set_ylabel("PDF", fontsize=30)
        ax.legend(fontsize=30)
        ax.grid(False)

    plt.tight_layout()
    filename = "error-pdf-compare"

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path_pdf = os.path.join(save_dir, filename + ".pdf")
        full_path_png = os.path.join(save_dir, filename + ".png")

        if out_pdf:
            with PdfPages(full_path_pdf) as pdf:
                pdf.savefig(fig, dpi=dpi)

        if out_png:
            fig.savefig(full_path_png, dpi=dpi)

    if out_show:
        plt.show()

    plt.close(fig)


def plot_pdf_comparison_to_pdf(errors_dict, pdf, dpi=600):
    """Create the same PDF comparison figure and append it to an open PdfPages.

    This mirrors plot_pdf_comparison but writes into the provided PdfPages
    instead of saving standalone files. Returns the PdfPages for chaining.
    """
    if not isinstance(errors_dict, dict) or not errors_dict:
        raise ValueError("errors_dict must be a non-empty dict of label -> 2D ndarray")

    first_key = next(iter(errors_dict))
    arr = errors_dict[first_key]
    if arr.ndim != 2:
        raise ValueError("Each errors_dict value must be a 2D array (N, D)")
    n_dims = arr.shape[1]

    fig, axs = plt.subplots(1, n_dims, figsize=(8 * n_dims, 8))
    if n_dims == 1:
        axs = [axs]

    mse_comp_label = [r'$\varepsilon_{u^{\prime}}$', r'$\varepsilon_{v^{\prime}}$']

    for dim in range(n_dims):
        ax = axs[dim]
        # Shared x-range estimation via Gaussian fit (kept for consistency)
        for label, data in errors_dict.items():
            data_dim = np.asarray(data)[:, dim]
            mu, std = norm.fit(data_dim)
            _ = np.linspace(mu - 4 * std, mu + 4 * std, 200)

        # KDE-based smooth PDFs
        for label, data in errors_dict.items():
            data_dim = np.asarray(data)[:, dim]
            kde = gaussian_kde(data_dim, bw_method=0.4)
            x_vals = np.linspace(np.min(data_dim), np.max(data_dim), 200)
            pdf_vals = kde.evaluate(x_vals)
            ax.plot(x_vals, pdf_vals, label=label, linewidth=2)
            ax.fill_between(x_vals, pdf_vals, alpha=0.2)
            ax.tick_params(axis='both', labelsize=30)
            ax.set_xlim(0, 0.035)
            ax.set_ylim(0, 400)

        ax.set_xlabel(mse_comp_label[dim] if dim < len(mse_comp_label) else f"dim {dim}", fontsize=50)
        ax.set_ylabel("PDF", fontsize=30)
        ax.legend(fontsize=30)
        ax.grid(False)

    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig, dpi=dpi)
    plt.close(fig)
    return pdf
