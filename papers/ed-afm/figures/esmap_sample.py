
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ppafm.ml.AuxMap as aux
import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu

from mlspm.utils import read_xyzs

if __name__ == "__main__":

    data_dir = Path("./edafm-data")  # Path to data
    save_dir = Path("./images/")  # Where images are saved

    scan_window = ((-8, -8), (8, 8))
    scan_dim = (128, 128)
    height = 4
    zmin = -2.0
    Rpp = 1.0

    # Initialize OpenCL environment on GPU
    env = oclu.OCLEnvironment(i_platform=0)
    FFcl.init(env)

    # Paths to molecule xyz file
    xyz_path = data_dir / "BCB" / "mol.xyz"

    # Define AuxMaps
    es_map = aux.ESMapConstant(scan_dim=scan_dim, scan_window=scan_window, height=height)
    vdw = aux.vdwSpheres(scan_dim=scan_dim, scan_window=scan_window, zmin=zmin, Rpp=Rpp)

    # Make sure save directory exists
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load molecule
    mol = read_xyzs([xyz_path])[0]
    xyzqs = mol[:, :4]
    xyzqs[:, :3] -= xyzqs[:, :3].mean(axis=0)
    Zs = mol[:, 4].astype(int)

    # Compute decriptors
    Y_es = es_map(xyzqs, Zs)
    Y_vdw = vdw(xyzqs, Zs)
    Y_vdw_mask = Y_vdw.copy()
    Y_vdw_mask -= Y_vdw_mask.min()
    Y_vdw_mask[Y_vdw_mask > 0.0] = 1.0
    Y_combined = Y_vdw_mask * Y_es

    # Plot ES field
    plt.figure(figsize=tuple(0.01 * np.array(Y_es.shape)), dpi=100)
    vmax = max(abs(Y_es.max()), abs(Y_es.min()))
    vmin = -vmax
    plt.imshow(Y_es.T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_dir / "es.png")
    plt.close()

    # Plot vdw spheres
    plt.figure(figsize=tuple(0.01 * np.array(Y_es.shape)), dpi=100)
    plt.imshow(Y_vdw.T, origin="lower", cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_dir / "vdw.png")
    plt.close()

    # Plot vdw mask
    plt.figure(figsize=tuple(0.01 * np.array(Y_es.shape)), dpi=100)
    plt.imshow(Y_vdw_mask.T, origin="lower", cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_dir / "vdw_mask.png")
    plt.close()

    # Plot combined
    plt.figure(figsize=tuple(0.01 * np.array(Y_es.shape)), dpi=100)
    vmax = max(abs(Y_combined.max()), abs(Y_combined.min()))
    vmin = -vmax
    plt.imshow(Y_combined.T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_dir / "es_cut.png")
    plt.close()
