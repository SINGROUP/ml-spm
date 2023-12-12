#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from torch import scatter
from ppafm.ocl.oclUtils import init_env

from plot_predictions import get_data as get_data_prediction, MM_TO_INCH
from plot_predictions import plot_graph as plot_graph_prediction
from plot_relaxed_structures import get_data as get_data_relaxed
from plot_relaxed_structures import plot_graph as plot_graph_relaxed

# Set matplotlib font rendering to use LaTex
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]})


def init_fig(width=140, left_margin=4, top_margin=4, row_gap=6, gap=0.5):
    ax_size = (width - left_margin - 5 * gap) / 5

    left_margin *= MM_TO_INCH
    top_margin *= MM_TO_INCH
    row_gap *= MM_TO_INCH
    gap *= MM_TO_INCH
    ax_size *= MM_TO_INCH
    width *= MM_TO_INCH
    height = top_margin + 2 * (ax_size + gap) + row_gap
    fig = plt.figure(figsize=(width, height))

    axes = []

    y = height - top_margin - ax_size
    x = left_margin
    axes_ = []
    for _ in range(5):
        rect = [x / width, y / height, ax_size / width, ax_size / height]
        ax = fig.add_axes(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0.5)
        axes_.append(ax)
        x += ax_size + gap
    axes.append(axes_)

    y = height - top_margin - 2 * ax_size - row_gap
    x = left_margin + 2 * (ax_size + gap)
    axes_ = []
    for _ in range(3):
        rect = [x / width, y / height, ax_size / width, ax_size / height]
        ax = fig.add_axes(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0.5)
        axes_.append(ax)
        x += ax_size + gap
    axes.append(axes_)

    return fig, axes


if __name__ == "__main__":
    init_env(i_platform=1)

    exp_data_dir = Path("./exp_data")
    sim_data_dir = Path("./relaxed_structures/")
    scatter_size = 5
    zmin = -5.0
    zmax = 0.5
    classes = [[1], [8], [29, 79]]
    class_colors = ["w", "r"]
    fontsize = 7

    params = {
        "pred_dir": "predictions_au111-bilayer",
        "sim_name": "hartree_I",
        "exp_name": "Ying_Jiang_4",
        "label": "I",
        "dist": 4.8,
        "rot_angle": -25.000,
        "amp": 2.0,
        "nz": 7,
        "offset": (0.0, 0.0),
    }

    exp_data, pred_mol, sim_pred = get_data_prediction(params, exp_data_dir, classes)
    opt_mol, sim_opt, _, sw_opt = get_data_relaxed(params, exp_data_dir, sim_data_dir, classes)

    fig, axes = init_fig()

    # Plot data
    axes[0][0].imshow(exp_data['data'][:, :, 0].T, origin="lower", cmap="gray")
    axes[0][1].imshow(exp_data['data'][:, :, -1].T, origin="lower", cmap="gray")
    plot_graph_prediction(
        axes[0][2],
        pred_mol,
        box_borders=[[0, 0, zmin], [exp_data["lengthX"], exp_data["lengthY"], zmax]],
        zmin=zmin,
        zmax=zmax,
        scatter_size=scatter_size,
        class_colors=class_colors,
    )
    axes[0][3].imshow(sim_pred[:, :, 0].T, origin="lower", cmap="gray")
    axes[0][4].imshow(sim_pred[:, :, -1].T, origin="lower", cmap="gray")
    plot_graph_relaxed(
        axes[1][0],
        opt_mol,
        box_borders=[[sw_opt[0][0], sw_opt[0][1], zmin], [sw_opt[1][0], sw_opt[1][1], zmax]],
        zmin=zmin,
        zmax=zmax,
        scatter_size=scatter_size,
        class_colors=class_colors,
    )
    axes[1][1].imshow(sim_opt[:, :, 0].T, origin="lower", cmap="gray")
    axes[1][2].imshow(sim_opt[:, :, -1].T, origin="lower", cmap="gray")

    # Set labels
    y = 1.08
    axes[0][0].text(
        -0.08, 0.5, params["label"], transform=axes[0][0].transAxes, fontsize=fontsize, va="center", ha="center", rotation="vertical"
    )
    axes[0][0].text(0.5, y, "Exp.\ AFM (far)", transform=axes[0][0].transAxes, fontsize=fontsize, va="center", ha="center")
    axes[0][1].text(0.5, y, "Exp.\ AFM (close)", transform=axes[0][1].transAxes, fontsize=fontsize, va="center", ha="center")
    axes[0][2].text(0.5, y, "Pred.\ geom.", transform=axes[0][2].transAxes, fontsize=fontsize, va="center", ha="center")
    axes[0][3].text(0.5, y, "Sim.\ AFM (far)", transform=axes[0][3].transAxes, fontsize=fontsize, va="center", ha="center")
    axes[0][4].text(0.5, y, "Sim.\ AFM (close)", transform=axes[0][4].transAxes, fontsize=fontsize, va="center", ha="center")
    axes[1][0].text(0.5, y, "Opt.\ geom.", transform=axes[1][0].transAxes, fontsize=fontsize, va="center", ha="center")
    axes[1][1].text(0.5, y, "Sim.\ AFM (far)", transform=axes[1][1].transAxes, fontsize=fontsize, va="center", ha="center")
    axes[1][2].text(0.5, y, "Sim.\ AFM (close)", transform=axes[1][2].transAxes, fontsize=fontsize, va="center", ha="center")

    plt.savefig(f"sims_extra.png", dpi=400)
