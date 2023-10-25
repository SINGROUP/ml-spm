import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, gridspec

from ..utils import _calc_plot_dim, elements
from . import MoleculeGraph

CLASS_COLORS = "rkbgcmy"


def plot_graphs( #TODO docstring
    pred: Optional[list[MoleculeGraph]] = None,
    ref: Optional[list[MoleculeGraph]] = None,
    box_borders: np.ndarray = np.array(((0, 0, -1.4), (16, 16, 0.5))),
    outdir: str = "./graphs/",
    classes: list[list[int]] = None,
    class_colors: list[str] = CLASS_COLORS,
    start_ind: int = 0,
    verbose: int = 1,
):
    n_plot = (pred is not None) + (ref is not None)
    if n_plot == 0:
        raise ValueError("pred and ref cannot both be None.")

    if (pred is not None) and (ref is not None) and (len(pred) != len(ref)):
        raise ValueError(f"pred ({len(pred)}) and ref ({len(ref)}) have different number of samples.")

    n_samples = len(pred) if pred is not None else len(ref)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if classes is None:
        atom_classes = []
        for mols in [pred, ref]:
            if mols is not None:
                for m in mols:
                    atom_classes += list(m.array(class_index=True))
        n_classes = max(atom_classes) + 1
        classes = [f"Class {i}" for i in range(n_classes)]
    else:
        n_classes = len(classes)
        classes = [", ".join([elements[e - 1] for e in c]) for c in classes]

    z_min = box_borders[0][2]
    z_max = box_borders[1][2]
    scatter_size = 160

    def get_marker_size(z, max_size):
        return max_size * (z - z_min) / (z_max - z_min)

    def plot_xy(ax, mol):
        if len(mol) > 0:
            mol_pos = mol.array(xyz=True)

            s = get_marker_size(mol_pos[:, 2], scatter_size)
            if (s < 0).any():
                raise ValueError("Encountered atom z position(s) below box borders.")

            c = [class_colors[atom.class_index] for atom in mol.atoms]

            ax.scatter(mol_pos[:, 0], mol_pos[:, 1], c=c, s=s, edgecolors="k", zorder=2)
            for b in mol.bonds:
                pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
                ax.plot(pos[:, 0], pos[:, 1], "k", linewidth=2, zorder=1)

        ax.set_xlim(box_borders[0][0], box_borders[1][0])
        ax.set_ylim(box_borders[0][1], box_borders[1][1])
        ax.set_aspect("equal", "box")

    def plot_xz(ax, mol):
        if len(mol) > 0:
            order = list(np.argsort(mol.array(xyz=True)[:, 1])[::-1])
            mol = mol.permute(order)
            mol_pos = mol.array(xyz=True)

            s = get_marker_size(mol_pos[:, 2], scatter_size)
            if (s < 0).any():
                raise ValueError("Encountered atom z position(s) below box borders.")

            c = [class_colors[atom.class_index] for atom in mol.atoms]

            for b in mol.bonds:
                pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
                ax.plot(pos[:, 0], pos[:, 2], "k", linewidth=2, zorder=1)
            ax.scatter(mol_pos[:, 0], mol_pos[:, 2], c=c, s=s, edgecolors="k", zorder=2)

        ax.set_xlim(box_borders[0][0], box_borders[1][0])
        ax.set_ylim(box_borders[0][2], box_borders[1][2])
        ax.set_aspect("equal", "box")

    ind = start_ind
    for i in range(n_samples):
        # Setup plot grid
        x_size = 5 * n_plot
        x_extra = 0.35 * max([len(c) for c in classes])
        fig = plt.figure(figsize=(x_size + x_extra, 6.5))
        fig_grid = gridspec.GridSpec(1, 2, width_ratios=(x_size, x_extra), wspace=1 / (x_size + x_extra))
        grid_graphs = fig_grid[0, 0].subgridspec(2, n_plot, height_ratios=(5, 1.5), hspace=0.1, wspace=0.2)

        # Prediction
        if pred is not None:
            ax_xy_pred = fig.add_subplot(grid_graphs[0, 0])
            ax_xz_pred = fig.add_subplot(grid_graphs[1, 0])
            plot_xy(ax_xy_pred, pred[i])
            plot_xz(ax_xz_pred, pred[i])
            ax_xy_pred.set_xlabel("x (Å)", fontsize=12)
            ax_xy_pred.set_ylabel("y (Å)", fontsize=12)
            ax_xz_pred.set_xlabel("x (Å)", fontsize=12)
            ax_xz_pred.set_ylabel("z (Å)", fontsize=12)
            ax_xy_pred.set_title("Prediction", fontsize=20)
            i_plot = 1
        else:
            i_plot = 0

        # Reference
        if ref is not None:
            ax_xy_ref = fig.add_subplot(grid_graphs[0, i_plot])
            ax_xz_ref = fig.add_subplot(grid_graphs[1, i_plot])
            plot_xy(ax_xy_ref, ref[i])
            plot_xz(ax_xz_ref, ref[i])
            ax_xy_ref.set_xlabel("x (Å)", fontsize=12)
            ax_xy_ref.set_ylabel("y (Å)", fontsize=12)
            ax_xz_ref.set_xlabel("x (Å)", fontsize=12)
            ax_xz_ref.set_ylabel("z (Å)", fontsize=12)
            ax_xy_ref.set_title("Reference", fontsize=20)

        # Plot legend
        ax_legend = fig.add_subplot(fig_grid[0, 1])

        # Class colors
        dy = 0.08
        dx = 0.35 / x_extra
        y_start = 0.5 + dy * (n_classes + 3) / 2
        for i, c in enumerate(classes):
            ax_legend.scatter(dx, y_start - dy * i, s=scatter_size, c=class_colors[i], edgecolors="k")
            ax_legend.text(2 * dx, y_start - dy * i, c, fontsize=16, ha="left", va="center_baseline")

        # Marker sizes
        y_start2 = y_start - (n_classes + 1) * dy
        marker_zs = np.array([z_max, (z_min + z_max + 0.2) / 2, z_min + 0.2])
        ss = get_marker_size(marker_zs, scatter_size)
        for i, (s, z) in enumerate(zip(ss, marker_zs)):
            ax_legend.scatter(dx, y_start2 - dy * i, s=s, c="w", edgecolors="k")
            ax_legend.text(2 * dx, y_start2 - dy * i, f"z = {z}Å", fontsize=16, ha="left", va="center_baseline")

        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.axis("off")

        plt.savefig(save_path := os.path.join(outdir, f"{ind}_graph.png"))
        if verbose > 0:
            print(f"Graph image saved to {save_path}")
        plt.close()

        ind += 1


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
