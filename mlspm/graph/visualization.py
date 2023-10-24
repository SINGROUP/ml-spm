import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from ..utils import _calc_plot_dim


def plot_distribution_grid(
    pred_dist: np.ndarray,
    ref_dist: Optional[np.ndarray] = None,
    box_borders: np.ndarray = np.array(((2, 2, -1.5), (18, 18, 0))),
    outdir: str = "./graphs/",
    start_ind: int = 0,
    verbose: int = 1,
):
    if ref_dist is not None:
        assert pred_dist.shape == ref_dist.shape, (pred_dist.shape, ref_dist.shape)

    n_img = 2 if ref_dist is not None else 1

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fontsize = 24

    z_start = box_borders[0][2]
    z_res = (box_borders[1][2] - box_borders[0][2]) / (pred_dist.shape[-1] - 1)
    extent = [box_borders[0][0], box_borders[1][0], box_borders[0][1], box_borders[1][1]]

    ind = start_ind
    for i in range(len(pred_dist)):
        p = pred_dist[i]
        r = ref_dist[i] if ref_dist is not None else None

        # Plot grid in 2D
        p_mean = p.mean(axis=-1)
        if r is not None:
            r_mean = r.mean(axis=-1)
            vmin = min(r_mean.min(), p_mean.min())
            vmax = max(r_mean.max(), p_mean.max())
        else:
            vmin, vmax = p_mean.min(), p_mean.max()
        fig, axes = plt.subplots(1, n_img, figsize=(2 + 5 * n_img, 6), squeeze=False)
        axes = axes[0]
        axes[0].imshow(p_mean.T, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
        axes[0].set_title("Prediction")
        if r is not None:
            axes[1].imshow(r_mean.T, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
            axes[1].set_title("Reference")

        # Colorbar
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        pos = axes[-1].get_position()
        cax = fig.add_axes(rect=[0.9, pos.ymin, 0.03, pos.ymax - pos.ymin])
        m = cm.ScalarMappable()
        m.set_array([vmin, vmax])
        plt.colorbar(m, cax=cax)

        plt.savefig(save_path := os.path.join(outdir, f"{ind}_pred_dist2D.png"))
        if verbose > 0:
            print(f"Position distribution 2D prediction image saved to {save_path}")
        plt.close()

        # Plot each z-slice separately
        if r is not None:
            vmin = min(r.min(), p.min())
            vmax = max(r.max(), p.max())
        else:
            vmin, vmax = p.min(), p.max()
        nrows, ncols = _calc_plot_dim(p.shape[-1], f=0.5)
        fig = plt.figure(figsize=(4 * ncols, 4.25 * nrows * n_img))
        fig_grid = fig.add_gridspec(nrows, ncols, wspace=0.05, hspace=0.15, left=0.03, right=0.98, bottom=0.02, top=0.98)
        for iz in range(p.shape[-1]):
            ix = iz % ncols
            iy = iz // ncols
            axes = fig_grid[iy, ix].subgridspec(n_img, 1, hspace=0.03).subplots(squeeze=False)[:, 0]
            axes[0].imshow(p[:, :, iz].T, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
            axes[0].axis("off")
            axes[0].set_title(f"z = {z_start + (iz + 0.5) * z_res:.2f}Å", fontsize=fontsize)
            if r is not None:
                axes[1].imshow(r[:, :, iz].T, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
                axes[1].axis("off")
            if ix == 0:
                axes[0].text(-0.1, 0.5, "Prediction", ha="center", va="center", transform=axes[0].transAxes, rotation="vertical", fontsize=fontsize)
                if r is not None:
                    axes[1].text(
                        -0.1, 0.5, "Reference", ha="center", va="center", transform=axes[1].transAxes, rotation="vertical", fontsize=fontsize
                    )

        plt.savefig(save_path := os.path.join(outdir, f"{ind}_pred_dist.png"))
        if verbose > 0:
            print(f"Position distribution prediction image saved to {save_path}")
        plt.close()

        ind += 1
