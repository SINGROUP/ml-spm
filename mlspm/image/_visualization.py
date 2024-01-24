from os import PathLike
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def make_prediction_plots(
    preds: List[np.ndarray] = None,
    true: List[np.ndarray] = None,
    losses: np.ndarray = None,
    descriptors: List[str] = None,
    outdir: PathLike = "./predictions/",
    start_ind: int = 0,
    verbose: bool = True,
):
    """
    Plot predictions/references for image descriptors.
    Arguments:
        preds: Predicted maps. Each list element corresponds to one descriptor and is an array of shape (batch_size, x_dim, y_dim).
        true: Reference maps. Each list element corresponds to one descriptor and is an array of shape (batch_size, x_dim, y_dim).
        losses: Losses for each prediction. Array of shape (len(preds), batch_size).
        descriptors: Names of descriptors. The name "ES" causes the coolwarm colormap to be used.
        outdir: Directory where images are saved.
        start_ind: Starting index for saved images.
        verbose: Whether to print output information.
    """

    rows = (preds is not None) + (true is not None)
    if rows == 0:
        raise ValueError("preds and true cannot both be None.")
    elif rows == 1:
        data = preds if preds is not None else true
    else:
        assert len(preds) == len(true)

    cols = len(preds) if preds is not None else len(true)
    if descriptors is not None:
        assert len(descriptors) == cols

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    img_ind = start_ind
    batch_size = len(preds[0]) if preds is not None else len(true[0])

    for j in range(batch_size):
        fig, axes = plt.subplots(rows, cols)
        fig.set_size_inches(6 * cols, 5 * rows)

        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for i in range(cols):
            top_ax = axes[0, i]
            bottom_ax = axes[-1, i]

            if rows == 2:
                p = preds[i][j]
                t = true[i][j]
                vmax = np.concatenate([p, t]).max()
                vmin = np.concatenate([p, t]).min()
            else:
                d = data[i][j]
                vmax = d.max()
                vmin = d.min()

            title1 = ""
            title2 = ""
            cmap = cm.viridis
            if descriptors is not None:
                descriptor = descriptors[i]
                title1 += f"{descriptor} Prediction"
                title2 += f"{descriptor} Reference"
                if descriptor == "ES":
                    vmax = max(abs(vmax), abs(vmin))
                    vmin = -vmax
                    cmap = cm.coolwarm
            if losses is not None:
                title1 += f"\nMSE = {losses[i,j]:.2E}"
            if vmax == vmin == 0:
                vmin = 0
                vmax = 0.1
            if rows == 2:
                im1 = top_ax.imshow(p.T, vmax=vmax, vmin=vmin, cmap=cmap, origin="lower")
                im2 = bottom_ax.imshow(t.T, vmax=vmax, vmin=vmin, cmap=cmap, origin="lower")
                if title1:
                    top_ax.set_title(title1)
                    bottom_ax.set_title(title2)
            else:
                im1 = top_ax.imshow(d.T, vmax=vmax, vmin=vmin, cmap=cmap, origin="lower")
                if title1:
                    title = title1 if preds is not None else title2
                    top_ax.set_title(title)

            for axi in axes[:, i]:
                pos = axi.get_position()
                pos_new = [pos.x0, pos.y0, 0.8 * (pos.x1 - pos.x0), pos.y1 - pos.y0]
                axi.set_position(pos_new)

            pos1 = top_ax.get_position()
            pos2 = bottom_ax.get_position()
            c_pos = [pos1.x1 + 0.1 * (pos1.x1 - pos1.x0), pos2.y0, 0.08 * (pos1.x1 - pos1.x0), pos1.y1 - pos2.y0]
            cbar_ax = fig.add_axes(c_pos)
            fig.colorbar(im1, cax=cbar_ax)

        save_name = outdir / f"{img_ind}_pred.png"
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()

        if verbose > 0:
            print(f"Prediction saved to {save_name}")
        img_ind += 1
