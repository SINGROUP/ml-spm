
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ppafm.ml.AuxMap as aux
import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu
import ppafm.ocl.relax as oclr
import torch
from matplotlib import cm
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl.AFMulator import AFMulator

import mlspm.preprocessing as pp
from mlspm.datasets import download_dataset
from mlspm.models import EDAFMNet

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

def apply_preprocessing_sim(batch):

    X, Y, xyzs = batch

    print(X[0].shape)

    X = [x[..., 2:8] for x in X]

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.08)

    # Add background gradient
    c = 0.3
    angle = -np.pi / 2
    x, y = np.meshgrid(np.arange(0, X[0].shape[1]), np.arange(0, X[0].shape[2]), indexing="ij")
    n = [np.cos(angle), np.sin(angle), 1]
    z = -(n[0]*x + n[1]*y)
    z -= z.mean()
    z /= np.ptp(z)
    for x in X:
        x += z[None, :, :, None]*c*np.ptp(x)

    return X, Y, xyzs

def apply_preprocessing_exp(X, real_dim):

    # Pick slices
    x0_start, x1_start = 2, 0
    X[0] = X[0][..., x0_start:x0_start+6] # CO
    X[1] = X[1][..., x1_start:x1_start+6] # Xe

    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = [x[:,:,6:78] for x in X]
    
    return X

if __name__ == "__main__":
    
    data_dir  = Path("./edafm-data") # Path to data
    X_slices  = [0, 3, 5]            # Which AFM slices to plot
    tip_names = ["CO", "Xe"]         # AFM tip types
    device    = "cuda"               # Device to run inference on
    fig_width = 140                  # Figure width in mm
    fontsize  = 8
    dpi       = 300

    # Download data if not already there
    download_dataset("ED-AFM-data", data_dir)

    # Initialize OpenCL environment on GPU
    env = oclu.OCLEnvironment( i_platform = 0 )
    FFcl.init(env)
    oclr.init(env)

    afmulator_args = {
        "pixPerAngstrome"   : 20,
        "scan_dim"          : (176, 144, 19),
        "scan_window"       : ((2.0, 2.0, 7.0), (24, 20, 8.9)),
        "df_steps"          : 10,
        "tipR0"             : [0.0, 0.0, 4.0]
    }

    generator_kwargs = {
        "batch_size"    : 1,
        "distAbove"     : 5.25,
        "iZPPs"         : [8, 54],
        "Qs"            : [[ -10, 20,  -10, 0 ], [  30, -60,   30, 0 ]],
        "QZs"           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
    }

    # Paths to molecule xyz files
    molecules = [data_dir / "PTCDA" / "mol.xyz"]

    # Define AFMulator
    afmulator = AFMulator(**afmulator_args)
    afmulator.npbc = (0,0,0)

    # Define AuxMaps
    aux_maps = [
        aux.ESMapConstant(
            scan_dim    = afmulator.scan_dim[:2],
            scan_window = [afmulator.scan_window[0][:2], afmulator.scan_window[1][:2]],
            height      = 4.0,
            vdW_cutoff  = -2.0,
            Rpp         = 1.0
        )
    ]

    # Define generator
    trainer = InverseAFMtrainer(afmulator, aux_maps, molecules, **generator_kwargs)

    # Get simulation data
    sim_data = next(iter(trainer))
    X_sim, ref, xyzs = apply_preprocessing_sim(sim_data)
    X_sim_cuda = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X_sim]

    # Load experimental data and preprocess
    data1 = np.load(data_dir / "PTCDA" / "data_CO_exp.npz")
    X1 = data1["data"]
    afm_dim1 = (data1["lengthX"], data1["lengthY"])

    data2 = np.load(data_dir / "PTCDA" / "data_Xe_exp.npz")
    X2 = data2["data"]
    afm_dim2 = (data2["lengthX"], data2["lengthY"])

    assert afm_dim1 == afm_dim2
    afm_dim = afm_dim1
    X_exp = apply_preprocessing_exp([X1[None], X2[None]], afm_dim)
    X_exp_cuda = [torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) for x in X_exp]

    # Load model with gradient augmentation
    model_grad = EDAFMNet(device=device, pretrained_weights="base")

    # Load model without gradient augmentation
    model_no_grad = EDAFMNet(device=device, pretrained_weights="no-gradient")

    with torch.no_grad():
        pred_sim_grad, attentions_sim_grad = model_grad(X_sim_cuda)
        pred_sim_no_grad, attentions_sim_no_grad = model_no_grad(X_sim_cuda)
        pred_exp, attentions_exp = model_no_grad(X_exp_cuda)
        pred_sim_grad = [p.cpu().numpy() for p in pred_sim_grad]
        pred_sim_no_grad = [p.cpu().numpy() for p in pred_sim_no_grad]
        pred_exp = [p.cpu().numpy() for p in pred_exp]
        attentions_sim_grad = [a.cpu().numpy() for a in attentions_sim_grad]
        attentions_sim_no_grad = [a.cpu().numpy() for a in attentions_sim_no_grad]
        attentions_exp = [a.cpu().numpy() for a in attentions_exp]

    # Create figure grid
    fig_width = 0.1/2.54*fig_width
    width_ratios = [6, 4.4]
    fig = plt.figure(figsize=(fig_width, 6*fig_width/sum(width_ratios)))
    fig_grid = fig.add_gridspec(1, 2, wspace=0.3, hspace=0, width_ratios=width_ratios)
    left_grid = fig_grid[0, 0].subgridspec(2, 1, wspace=0, hspace=0.1)

    pred_sim_grid = fig_grid[0, 1].subgridspec(2, 1, wspace=0, hspace=0.1)
    pred_sim_no_grad_ax, cbar_sim_no_grad_ax = pred_sim_grid[0, 0].subgridspec(1, 2, wspace=0.05,
        hspace=0, width_ratios=[1, 0.08]).subplots()
    pred_sim_grad_ax, cbar_sim_grad_ax = pred_sim_grid[1, 0].subgridspec(1, 2, wspace=0.05,
        hspace=0, width_ratios=[1, 0.08]).subplots()
    pred_exp_ax, cbar_exp_ax = left_grid[0, 0].subgridspec(1, 2, wspace=0.05, width_ratios=[1, 0.05]).subplots()
    afm_axes = left_grid[1, 0].subgridspec(len(X_sim), len(X_slices), wspace=0.01, hspace=0.01).subplots(squeeze=False)

    # Plot AFM
    for i, x in enumerate(X_sim):
        for j, s in enumerate(X_slices):
            
            # Plot AFM slice
            im = afm_axes[i, j].imshow(x[0,:,:,s].T, origin="lower", cmap="afmhot")
            afm_axes[i, j].set_axis_off()
        
        # Put tip names to the left of the AFM image rows
        afm_axes[i, 0].text(-0.1, 0.5, tip_names[i], horizontalalignment="center",
            verticalalignment="center", transform=afm_axes[i, 0].transAxes,
            rotation="vertical", fontsize=fontsize)

    # Figure out ES data limits
    vmax_sim_no_grad = max(abs(pred_sim_no_grad[0].min()), abs(pred_sim_no_grad[0].max()))
    vmax_sim_grad = max(abs(pred_sim_grad[0].min()), abs(pred_sim_grad[0].max()))
    vmax_exp = max(abs(pred_exp[0].min()), abs(pred_exp[0].max()))
    vmin_sim_no_grad = -vmax_sim_no_grad
    vmin_sim_grad = -vmax_sim_grad
    vmin_exp = -vmax_exp

    # Plot ES predictions
    pred_sim_no_grad_ax.imshow(pred_sim_no_grad[0][0].T, origin="lower", cmap="coolwarm",
        vmin=vmin_sim_no_grad, vmax=vmax_sim_no_grad)
    pred_sim_grad_ax.imshow(pred_sim_grad[0][0].T, origin="lower", cmap="coolwarm",
        vmin=vmin_sim_grad, vmax=vmax_sim_grad)
    pred_exp_ax.imshow(pred_exp[0][0].T, origin="lower", cmap="coolwarm", vmin=vmin_exp, vmax=vmax_exp)

    pred_sim_no_grad_ax.set_axis_off()
    pred_sim_grad_ax.set_axis_off()
    pred_exp_ax.set_axis_off()

    # Plot ES Map colorbar for no grad prediction
    m_es = cm.ScalarMappable(cmap=cm.coolwarm)
    m_es.set_array((vmin_sim_no_grad, vmax_sim_no_grad))
    cbar = plt.colorbar(m_es, cax=cbar_sim_no_grad_ax)
    cbar.set_ticks([-0.1, 0.0, 0.1])
    cbar_sim_no_grad_ax.tick_params(labelsize=fontsize-1)
    cbar.set_label("V/Å", fontsize=fontsize)

    # Plot ES Map colorbar for grad prediction
    m_es = cm.ScalarMappable(cmap=cm.coolwarm)
    m_es.set_array((vmin_sim_grad, vmax_sim_grad))
    cbar = plt.colorbar(m_es, cax=cbar_sim_grad_ax)
    cbar.set_ticks([-0.1, 0.0, 0.1])
    cbar_sim_grad_ax.tick_params(labelsize=fontsize-1)
    cbar.set_label("V/Å", fontsize=fontsize)

    # Plot ES Map colorbar for experimental prediction
    m_es = cm.ScalarMappable(cmap=cm.coolwarm)
    m_es.set_array((vmin_exp, vmax_exp))
    cbar = plt.colorbar(m_es, cax=cbar_exp_ax)
    cbar.set_ticks([-0.04, 0.0, 0.04])
    cbar_exp_ax.tick_params(labelsize=fontsize-1)
    cbar.set_label("V/Å", fontsize=fontsize)

    # Set labels
    pred_exp_ax.text(-0.06, 0.98, "A", horizontalalignment="center",
        verticalalignment="center", transform=pred_exp_ax.transAxes, fontsize=fontsize)
    afm_axes[0, 0].text(-0.2, 1.0, "B", horizontalalignment="center",
        verticalalignment="center", transform=afm_axes[0, 0].transAxes, fontsize=fontsize)
    pred_sim_no_grad_ax.text(-0.08, 0.98, "C", horizontalalignment="center",
        verticalalignment="center", transform=pred_sim_no_grad_ax.transAxes, fontsize=fontsize)
    pred_sim_grad_ax.text(-0.08, 0.98, "D", horizontalalignment="center",
        verticalalignment="center", transform=pred_sim_grad_ax.transAxes, fontsize=fontsize)

    plt.savefig("background_gradient.pdf", bbox_inches="tight", dpi=dpi)