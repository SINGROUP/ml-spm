from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlspm.datasets import download_dataset

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

if __name__ == "__main__":
    data_dir = Path("./edafm-data")
    fig_width = 160
    fontsize = 8
    dpi = 300

    # Download data if not already there
    download_dataset("ED-AFM-data", data_dir)

    # Load data
    bcb_CO = np.load(data_dir / "BCB" / "data_CO_exp.npz")
    bcb_Xe = np.load(data_dir / "BCB" / "data_Xe_exp.npz")
    ptcda_CO = np.load(data_dir / "PTCDA" / "data_CO_exp.npz")
    ptcda_Xe = np.load(data_dir / "PTCDA" / "data_Xe_exp.npz")

    fig_width = 0.1 / 2.54 * fig_width
    height_ratios = [2, 2, 2.45, 2.45]
    fig = plt.figure(figsize=(fig_width, 0.85 * sum(height_ratios)))
    fig_grid = fig.add_gridspec(4, 1, wspace=0, hspace=0.1, height_ratios=height_ratios)

    # BCB plots
    for i, (sample, label) in enumerate(zip([bcb_CO, bcb_Xe], ["A", "B"])):
        d = sample["data"]
        l = sample["lengthX"]
        axes = fig_grid[i, 0].subgridspec(2, 8, wspace=0.02, hspace=0.02).subplots().flatten()
        for j, ax in enumerate(axes):
            if j < d.shape[-1]:
                ax.imshow(d[:, :, j].T, origin="lower", cmap="afmhot")
            ax.axis("off")
        axes[0].text(
            -0.3, 0.8, label, horizontalalignment="center", verticalalignment="center", transform=axes[0].transAxes, fontsize=fontsize
        )
        axes[0].plot([50, 50 + 5 / l * d.shape[0]], [470, 470], color="k")

    # PTCDA plots
    for i, (sample, label) in enumerate(zip([ptcda_CO, ptcda_Xe], ["C", "D"])):
        d = sample["data"]
        l = sample["lengthX"]
        axes = fig_grid[i + 2, 0].subgridspec(3, 6, wspace=0.02, hspace=0.02).subplots().flatten()
        for j, ax in enumerate(axes):
            if j < d.shape[-1]:
                ax.imshow(d[:, :, j].T, origin="lower", cmap="afmhot")
            ax.axis("off")
        axes[0].text(
            -0.22, 0.7, label, horizontalalignment="center", verticalalignment="center", transform=axes[0].transAxes, fontsize=fontsize
        )
        axes[0].plot([20, 20 + 5 / l * d.shape[0]], [135, 135], color="k")

    plt.savefig("afm_stacks.pdf", bbox_inches="tight", dpi=dpi)
