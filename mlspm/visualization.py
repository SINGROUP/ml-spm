import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from . import common_utils as cu
# Make subpackage plotting tools also available
from .graph.visualization import *


def plot_input(X: np.ndarray, constant_range: bool = False, cmap: str | Colormap = "afmhot") -> plt.Figure:
    """
    Plot a single stack of AFM images.

    Arguments:
        X: AFM image to plot.
        constant_range: Whether the different slices should use the same value range or not.
        cmap: Colormap to use for plotting.

    Returns: Figure on which the image was plotted.
    """
    rows, cols = cu._calc_plot_dim(X.shape[-1])
    fig = plt.figure(figsize=(3.2 * cols, 2.5 * rows))
    vmax = X.max()
    vmin = X.min()
    for k in range(X.shape[-1]):
        fig.add_subplot(rows, cols, k + 1)
        if constant_range:
            plt.imshow(X[:, :, k].T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        else:
            plt.imshow(X[:, :, k].T, cmap=cmap, origin="lower")
        plt.colorbar()
    plt.tight_layout()
    return fig


def make_input_plots(
    Xs: list[np.ndarray],
    outdir: str = "./predictions/",
    start_ind: int = 0,
    constant_range: bool = False,
    cmap: str | Colormap = "afmhot",
    verbose: int = 1,
):
    """
    Plot a batch of AFM images to files 0_input.png, 1_input.png, ... etc.

    Arguments:
        Xs: Input AFM images to plot. Each list element corresponds to one AFM tip and is an array of shape (batch, x, y, z).
        outdir: Directory where images are saved.
        start_ind: Save index increments by one for each image. The first index is start_ind.
        constant_range: Whether the different slices should use the same value range or not.
        cmap: Colormap to use for plotting.
        verbose: Whether to print output information.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    img_ind = start_ind
    for i in range(Xs[0].shape[0]):
        for j in range(len(Xs)):
            plot_input(Xs[j][i], constant_range, cmap=cmap)

            save_name = f"{img_ind}_input"
            if len(Xs) > 1:
                save_name += str(j + 1)
            save_name = os.path.join(outdir, save_name)
            save_name += ".png"
            plt.savefig(save_name)
            plt.close()

            if verbose > 0:
                print(f"Input image saved to {save_name}")

        img_ind += 1
