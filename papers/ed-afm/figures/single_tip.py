
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
from mlspm.models import EDAFMNet

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

def apply_preprocessing_sim(batch):

    X, Y, xyzs = batch

    X = [x[..., 2:8] for x in X]

    pp.add_norm(X)
    np.random.seed(0)
    pp.add_noise(X, c=0.08)

    return X, Y, xyzs

def apply_preprocessing_bcb(X, real_dim):
    x0_start = 4
    X[0] = X[0][..., x0_start:x0_start+6] # CO
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    return X

def apply_preprocessing_ptcda(X, real_dim):
    x0_start = 2
    X[0] = X[0][..., x0_start:x0_start+6] # CO
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = [x[:,:,6:78] for x in X]
    return X

if __name__ == "__main__":

    data_dir    = Path("./edafm-data")  # Path to data
    device      = "cuda"                # Device to run inference on
    fig_width   = 160                   # Figure width in mm
    fontsize    = 8
    dpi         = 300

    # Initialize OpenCL environment on GPU
    env = oclu.OCLEnvironment( i_platform = 0 )
    FFcl.init(env)
    oclr.init(env)

    afmulator_args = {
        "pixPerAngstrome"   : 20,
        "scan_dim"          : (128, 128, 19),
        "scan_window"       : ((2.0, 2.0, 7.0), (20.0, 20.0, 8.9)),
        "df_steps"          : 10,
        "tipR0"             : [0.0, 0.0, 4.0]
    }

    generator_kwargs = {
        "batch_size"    : 1,
        "distAbove"     : 5.25,
        "iZPPs"         : [8],
        "Qs"            : [[ -10, 20,  -10, 0 ]],
        "QZs"           : [[ 0.1,  0, -0.1, 0 ]]
    }

    # Paths to molecule xyz files
    molecules = [data_dir / "TTF-TDZ" / "mol.xyz"]

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
    batch = next(iter(trainer))
    X_sim, Y_sim, xyz = apply_preprocessing_sim(batch)
    X_sim_cuda = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X_sim]

    # Load BCB data and preprocess
    data_bcb = np.load(data_dir / "BCB" / "data_CO_exp.npz")
    X_bcb = data_bcb["data"]
    afm_dim_bcb = (data_bcb["lengthX"], data_bcb["lengthY"])
    X_bcb = apply_preprocessing_bcb([X_bcb[None]], afm_dim_bcb)
    X_bcb_cuda = [torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) for x in X_bcb]

    # Load PTCDA data and preprocess
    data_ptcda = np.load(data_dir / "PTCDA" / "data_CO_exp.npz")
    X_ptcda = data_ptcda["data"]
    afm_dim_ptcda = (data_ptcda["lengthX"], data_ptcda["lengthY"])
    X_ptcda = apply_preprocessing_ptcda([X_ptcda[None]], afm_dim_ptcda)
    X_ptcda_cuda = [torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) for x in X_ptcda]

    # Load model
    model = EDAFMNet(device=device, pretrained_weights="single-channel")

    # Make predictions
    with torch.no_grad():
        pred_sim, attentions_sim = model(X_sim_cuda)
        pred_bcb, attentions_bcb = model(X_bcb_cuda)
        pred_ptcda, attentions_ptcda = model(X_ptcda_cuda)
        pred_sim = [p.cpu().numpy() for p in pred_sim]
        pred_bcb = [p.cpu().numpy() for p in pred_bcb]
        pred_ptcda = [p.cpu().numpy() for p in pred_ptcda]
        attentions_sim = [a.cpu().numpy() for a in attentions_sim]
        attentions_bcb = [a.cpu().numpy() for a in attentions_bcb]
        attentions_ptcda = [a.cpu().numpy() for a in attentions_ptcda]

    # Make figure
    fig_width = 0.1/2.54*fig_width
    width_ratios = [4, 4, 6.9]
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 2.6*fig_width/sum(width_ratios)),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.3})

    tick_arrays = [
        [-0.03, 0.0, 0.03],
        [-0.05, 0.0, 0.05],
        [-0.1, 0.0, 0.1]
    ]

    # Plot all predictions
    for ax, ticks, pred, label in zip(axes, tick_arrays, [pred_sim, pred_bcb, pred_ptcda], ["A", "B", "C"]):
        vmax = max(abs(pred[0][0].max()), abs(pred[0][0].min())); vmin = -vmax
        ax.imshow(pred[0][0].T, vmin=vmin, vmax=vmax, cmap="coolwarm", origin="lower")
        plt.axes(ax)
        m = cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array([vmin, vmax])
        cbar = plt.colorbar(m, ax=ax)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=fontsize-1)
        cbar.set_label("V/Å", fontsize=fontsize)
        ax.text(-0.1, 0.95, label, horizontalalignment="center",
            verticalalignment="center", transform=ax.transAxes, fontsize=fontsize)
        ax.set_axis_off()

    # Calculate relative error metric for simulation
    rel_abs_err_es = np.mean(np.abs(pred_sim[0] - Y_sim[0])) / np.ptp(Y_sim[0])
    print(f"Relative error: {rel_abs_err_es*100:.2f}%")

    plt.savefig("single_tip_predictions.pdf", bbox_inches="tight", dpi=dpi)
