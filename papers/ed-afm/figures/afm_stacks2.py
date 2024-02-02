
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
    fig_width   = 160
    fontsize    = 8
    dpi         = 300

    # Download data if not already there
    download_dataset("ED-AFM-data", data_dir)

    # Load data
    water_CO = np.load(data_dir / 'Water' / 'data_CO_exp.npz')
    water_Xe = np.load(data_dir / 'Water' / 'data_Xe_exp.npz')

    fig = plt.figure(figsize=(0.1/2.54*fig_width, 5.0))
    fig_grid = fig.add_gridspec(2, 1, wspace=0, hspace=0.1)

    # Water plots
    for i, (sample, label) in enumerate(zip([water_CO, water_Xe], ['E', 'F'])):
        d = sample['data']
        l = sample['lengthX']
        axes = fig_grid[i, 0].subgridspec(3, 8, wspace=0.02, hspace=0.02).subplots().flatten()
        for j, ax in enumerate(axes):
            if j < d.shape[-1]:
                ax.imshow(d[:,:,j].T, origin='lower', cmap='afmhot')
            ax.axis('off')
        axes[0].text(-0.3, 0.8, label, horizontalalignment='center',
            verticalalignment='center', transform=axes[0].transAxes, fontsize=fontsize)
        axes[0].plot([50, 50+5/l*d.shape[0]], [470, 470], color='k')

    plt.savefig('afm_stacks2.pdf', bbox_inches='tight', dpi=dpi)
