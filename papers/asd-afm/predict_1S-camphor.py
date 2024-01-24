#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from mlspm.datasets import download_dataset

from mlspm.models import ASDAFMNet
from mlspm.visualization import make_input_plots, make_prediction_plots

def plot_experiment_AFM(list_of_exp):
    cols = 10
    rows = len(list_of_exp)
    fig = plt.figure(figsize=(1.8 * cols, 1.8 * rows))
    ax = []
    for i, experiment in enumerate(list_of_exp):
        filename_exp = data_dir / "experimental_configs_data/" / f"{experiment}orient_exp.npz"
        data = np.load(filename_exp)
        X_exp = data["X"]
        print("#" + str(experiment) + " 1S-Champhor experiment " + "X.shape:", X_exp.shape)

        for j in range(10):
            ax.append(fig.add_subplot(rows, cols, i * cols + j + 1))
            xj = X_exp[0, :, :, j]
            vmax = xj.max()
            vmin = xj.min()
            plt.imshow(xj, cmap="afmhot", origin="lower", vmin=vmin - 0.1 * (vmax - vmin), vmax=vmax + 0.1 * (vmax - vmin))
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                ax[-1].set_ylabel("AFM experiment " + str(experiment))

    plt.show()



def plot_experiment_preds(list_of_exp):
    cols = len(list_of_exp)
    rows = 1
    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    ax = []
    for i, experiment in enumerate(list_of_exp):
        filename_exp = data_dir / "experimental_configs_data/" / f"{experiment}orient_exp.npz"
        data = np.load(filename_exp)
        Y_exp = data["Y"]
        ax.append(fig.add_subplot(rows, cols, i + 1))
        plt.imshow(Y_exp[1][0], origin="lower")
        ax[-1].set_ylabel("vdW-Spheres")
        ax[-1].set_xlabel("AFM experiment " + str(experiment))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':

    # Input data path
    data_dir = Path("afm_camphor")

    # Output directory
    outdir = Path('predictions')
    outdir.mkdir(exist_ok=True)

    # Type of pretrained weights to load ('light' or 'heavy')
    weights = 'heavy'

    # Device to run inference on. Set to 'cuda' to use GPU-acceleration.
    device = 'cpu'

    # Descriptor labels for plotting
    descriptors = ['Atomic Disks', 'vdW Spheres', 'Height Map']

    # Load model with pretrained weights
    model = ASDAFMNet(pretrained_weights=f'asdafm-{weights}').to(device)

    # Download AFM data
    download_dataset('AFM-camphor-exp', data_dir)

    for exp_num in [1, 3, 4, 6, 7]:

        # Load data
        X = np.load(data_dir / f'{exp_num}.npy')
        X = torch.from_numpy(X).float().unsqueeze(1).to(device)

        # Run prediction
        with torch.no_grad():
            pred = model(X)
            pred = [p.cpu().numpy() for p in pred]

        # The input data here is saved in a transposed form (y, x). Transpose it to x, y order so that the plotting
        # utils work correctly.
        pred = [p.transpose(0, 2, 1) for p in pred]
        X = [x.squeeze(1).cpu().numpy().transpose(0, 2, 1, 3) for x in X]

        # Plot the inputs (AFM) and predictions (descriptors)
        make_input_plots(X, outdir=outdir, start_ind=exp_num)
        make_prediction_plots(pred, descriptors=descriptors, outdir=outdir, start_ind=exp_num)
