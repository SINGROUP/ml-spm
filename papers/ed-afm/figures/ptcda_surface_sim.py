
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

import mlspm.preprocessing as pp
from mlspm.models import EDAFMNet

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

def apply_preprocessing(batch):

    X, Y, xyzs = batch

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.08)

    return X, Y, xyzs

if __name__ == "__main__":

    data_dir    = Path("./edafm-data")  # Path to data
    X_slices    = [0, 3, 5]             # Which AFM slices to plot
    tip_names   = ["CO", "Xe"]          # AFM tip types
    device      = "cuda"                # Device to run inference on
    fig_width   = 150                   # Figure width in mm
    fontsize    = 8
    dpi         = 300

    # Load model
    model = EDAFMNet(device=device, pretrained_weights="base")

    # Loop over molecules and plot
    fig_width = 0.1/2.54*fig_width
    width_ratios = [6, 8, 0.3]
    fig = plt.figure(figsize=(fig_width, 2.88*fig_width/sum(width_ratios)))

    # Define ticks for colorbars
    ticks = [-0.08, -0.04, 0.00, 0.04, 0.08]

    # Load data
    X1 = np.load(data_dir / "PTCDA" / "data_CO_sim.npz")["data"].astype(np.float32)
    X2 = np.load(data_dir / "PTCDA" / "data_Xe_sim.npz")["data"].astype(np.float32)
    X = [X1[None], X2[None]]
    Y_hartree = [np.load(data_dir / "PTCDA" / "ESMapHartree.npy")[None]]

    X, Y_hartree, _ = apply_preprocessing((X, Y_hartree, None))
    with torch.no_grad():
        X = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X]
        pred, attentions = model(X)
        pred = [p.cpu().numpy() for p in pred]
        attentions = [a.cpu().numpy() for a in attentions]
        X = [x.squeeze(1).cpu().numpy() for x in X]

    # Create plot grid
    sample_grid = fig.add_gridspec(1, 3, wspace=0.01, hspace=0, width_ratios=width_ratios)
    input_axes = sample_grid[0, 0].subgridspec(len(X), len(X_slices), wspace=0.01, hspace=0.02).subplots(squeeze=False)
    pred_ax, ref_ax = sample_grid[0, 1].subgridspec(1, 2, wspace=0.01, hspace=0).subplots(squeeze=True)
    cbar_ax = sample_grid[0, 2].subgridspec(1, 1, wspace=0, hspace=0).subplots(squeeze=True)

    # Plot AFM inputs
    ims = []
    for i, x in enumerate(X):
        for j, s in enumerate(X_slices):
            ims.append(input_axes[i, j].imshow(x[0,:,:,s].T, origin="lower", cmap="afmhot"))
            input_axes[i, j].set_axis_off()
        input_axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment="center",
            verticalalignment="center", transform=input_axes[i, 0].transAxes,
            rotation="vertical", fontsize=fontsize)

    # Figure out data limits
    vmax = max(
        abs(pred[0].min()), abs(pred[0].max()),
        abs(Y_hartree[0][0].min()), abs(Y_hartree[0][0].max())
    )
    vmin = -vmax

    # Plot prediction and references
    pred_ax.imshow(pred[0][0].T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ref_ax.imshow(Y_hartree[0][0].T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)

    # Plot ES Map colorbar
    plt.rcParams["font.serif"] = "cmr10"
    m_es = cm.ScalarMappable(cmap=cm.coolwarm)
    m_es.set_array((vmin, vmax))
    cbar = plt.colorbar(m_es, cax=cbar_ax)
    cbar.set_ticks(ticks)
    cbar_ax.tick_params(labelsize=fontsize-1)
    cbar.set_label("V/Å", fontsize=fontsize)

    # Turn off axes ticks
    pred_ax.set_axis_off()
    ref_ax.set_axis_off()

    # Set titles
    input_axes[0, len(X_slices)//2].set_title("AFM simulation (Hartree)", fontsize=fontsize, y=0.93)
    pred_ax.set_title("Prediction", fontsize=fontsize, y=0.97)
    ref_ax.set_title("Reference (Hartree)", fontsize=fontsize, y=0.97)

    plt.savefig("surface_sims_ptcda.pdf", bbox_inches="tight", dpi=dpi)
