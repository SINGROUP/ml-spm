#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ppafm.ocl.AFMulator import AFMulator

from mlspm.graph import MoleculeGraph
from mlspm.utils import read_xyzs

# # Set matplotlib font rendering to use LaTex
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

MM_TO_INCH = 1 / 25.4

def get_sim(mol, exp_data, params):

    amp = 2.0
    nx, ny, nz = exp_data['data'].shape
    dist = params['dist']

    xyzs = mol.array(xyz=True)
    Zs = mol.array(element=True).astype(np.int32)[:, 0]
    qs = np.zeros(len(xyzs))

    df_steps = round(amp / 0.1)
    scan_dim = (nx, ny, df_steps + nz - 1,)
    zmin = xyzs[:, 2].max() + dist
    zmax = zmin + (amp - 0.1) + (nz - 1) * 0.1
    scan_window = ((0.01, 0.01, zmin), (exp_data['lengthX'], exp_data['lengthY'], zmax))

    afmulator = AFMulator(
        pixPerAngstrome=10,
        scan_dim=scan_dim,
        scan_window=scan_window,
        iZPP=8,
        df_steps=df_steps
    )

    X = afmulator(xyzs, Zs, qs)

    return X

def crop_data(exp_data, pred_mol, sim):

    xyzs = pred_mol.array(xyz=True)
    xyzs_min = xyzs.min(axis=0)[:2] - 5.0
    xyzs_max = xyzs.max(axis=0)[:2] + 5.0

    x_res = exp_data['lengthX'] / sim.shape[0]
    y_res = exp_data['lengthY'] / sim.shape[1]
    ratio = sim.shape[0] / sim.shape[1]

    # Try to get close to a square shape with the system in the middle
    if ratio >= 1.0:
        x_start = max(round(xyzs_min[0] / x_res), 0)
        x_end = min(round(xyzs_max[0] / x_res), sim.shape[0])
        x_size = x_end - x_start
        y_middle = round((xyzs_max[1] + xyzs_min[1]) / 2 / y_res)
        y_start = max(0, y_middle - x_size // 2)
        y_end = min(y_start + x_size, sim.shape[1])
    else:
        y_start = max(round(xyzs_min[1] / y_res), 0)
        y_end = min(round(xyzs_max[1] / y_res), sim.shape[1])
        y_size = y_end - y_start
        x_middle = round((xyzs_max[0] + xyzs_min[0]) / 2 / x_res)
        x_start = max(0, x_middle - y_size // 2)
        x_end = min(x_start + y_size, sim.shape[0])

    x_shift = x_start * x_res
    y_shift = y_start * y_res

    pred_mol = pred_mol.transform_xy(shift=(-x_shift, -y_shift))

    exp_data = {
        'data': exp_data['data'][x_start:x_end, y_start:y_end],
        'lengthX': (x_end - x_start) * x_res,
        'lengthY': (y_end - y_start) * y_res

    }
    sim = sim[x_start:x_end, y_start:y_end]

    return exp_data, pred_mol, sim

def get_data(params, exp_data_dir, classes):

    print('Loading', params['label'])

    exp_data = np.load(Path(exp_data_dir) / f"{params['exp_name']}.npz")
    exp_data = {
        'data': exp_data['data'][..., -15:],
        'lengthX': exp_data['lengthX'],
        'lengthY': exp_data['lengthY'],
    }

    pred_dir = Path(params['pred_dir'])
    pred_xyz = read_xyzs([pred_dir / f"{params['exp_name']}_mol.xyz"])[0]
    pred_bonds = np.loadtxt(pred_dir / f"{params['exp_name']}_bonds.txt", delimiter=' ', dtype=int)
    pred_mol = MoleculeGraph(pred_xyz, pred_bonds, classes=classes)

    sim = get_sim(pred_mol, exp_data, params)
    exp_data, pred_mol, sim = crop_data(exp_data, pred_mol, sim)

    return exp_data, pred_mol, sim

def init_fig(data, height=200, gap=0.5, left_margin=4, top_margin=4, num_cols=6, third_row_skip=1.5):

    effective_height = height - top_margin - len(data) * gap - third_row_skip
    img_ratios = [(d[0]['data'].shape[1] / d[0]['data'].shape[0]) for d in data]
    ax_width = effective_height / sum(img_ratios) * MM_TO_INCH
    ax_heights = [ax_width * r for r in img_ratios]

    left_margin *= MM_TO_INCH
    top_margin *= MM_TO_INCH
    gap *= MM_TO_INCH
    height *= MM_TO_INCH
    third_row_skip *= MM_TO_INCH
    width = left_margin + num_cols * gap + num_cols * ax_width
    fig = plt.figure(figsize=(width, height))

    axes = []
    y = height - top_margin
    for i_row, h in enumerate(ax_heights):

        x = left_margin
        y -= h
        axes_ = []
        for _ in range(num_cols):
            rect = [x / width, y / height, ax_width / width, h / height]
            ax = fig.add_axes(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.5)
            axes_.append(ax)
            x += (ax_width + gap)
        axes.append(axes_)

        if i_row == 1:
            y -= third_row_skip
            line_width = num_cols * (ax_width + gap) - gap
            rect = [left_margin / width, y / height, line_width / width, (third_row_skip - gap) / height]
            ax = fig.add_axes(rect)
            ax.hlines(y=0, xmin=-1, xmax=1, linestyles='dashed', colors='black', linewidth=0.8)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()

        y -= gap

    return fig, axes

def get_marker_size(z, scatter_size, zmin, zmax):
    return scatter_size * (z - zmin) / (zmax - zmin)

def plot_graph(ax, mol, box_borders, class_colors, scatter_size, zmin, zmax):

    mol_pos = mol.array(xyz=True)

    s = get_marker_size(mol_pos[:, 2], scatter_size, zmin, zmax)
    if (s < 0).any():
        print(mol_pos[:, 2].min())
        raise ValueError('Encountered atom z position(s) below box borders.')
    
    c = [class_colors[atom.class_index] for atom in mol.atoms]

    ax.scatter(mol_pos[:,0], mol_pos[:,1], c=c, s=s, edgecolors='k', zorder=2,
        linewidths=0.15)
    for b in mol.bonds:
        pos = np.vstack([mol_pos[b[0]], mol_pos[b[1]]])
        ax.plot(pos[:,0], pos[:,1], 'k', linewidth=0.15, zorder=1)
    
    ax.set_xlim(box_borders[0][0], box_borders[1][0])
    ax.set_ylim(box_borders[0][1], box_borders[1][1])
    ax.set_aspect('equal', 'box')

if __name__ == '__main__':

    exp_data_dir = Path('./exp_data/')
    scatter_size = 5
    zmin = -3.5
    zmax = 0.5
    classes = [[1], [8]]
    class_colors = ['w', 'r']
    fontsize = 7

    params = [
        {'pred_dir': 'predictions_au111-monolayer', 'exp_name': 'Ying_Jiang_7'  , 'label': 'A', 'dist': 5.5, 'offset': ( 1.0,  0.0)},
        {'pred_dir': 'predictions_cu111'          , 'exp_name': 'Chen_CO'       , 'label': 'B', 'dist': 5.5, 'offset': (-1.0,  0.5)},
        {'pred_dir': 'predictions_au111-bilayer'  , 'exp_name': 'Ying_Jiang_1'  , 'label': 'C', 'dist': 5.1, 'offset': ( 0.0, -1.0)},
        {'pred_dir': 'predictions_au111-bilayer'  , 'exp_name': 'Ying_Jiang_2_1', 'label': 'D', 'dist': 5.0, 'offset': ( 1.0,  0.5)},
        {'pred_dir': 'predictions_au111-bilayer'  , 'exp_name': 'Ying_Jiang_2_2', 'label': 'E', 'dist': 4.9, 'offset': ( 0.0,  0.0)},
        {'pred_dir': 'predictions_au111-bilayer'  , 'exp_name': 'Ying_Jiang_3'  , 'label': 'F', 'dist': 4.8, 'offset': ( 0.0, -2.0)},
        {'pred_dir': 'predictions_au111-bilayer'  , 'exp_name': 'Ying_Jiang_5'  , 'label': 'G', 'dist': 5.0, 'offset': ( 2.0,  0.0)},
        {'pred_dir': 'predictions_au111-bilayer'  , 'exp_name': 'Ying_Jiang_6'  , 'label': 'H', 'dist': 4.8, 'offset': ( 1.5,  2.0)}
    ]

    data = [get_data(p, exp_data_dir, classes) for p in params]
    fig, axes = init_fig(data, height=180, num_cols=5)

    for i, (row_axes, (exp_data, pred_mol, sim)) in enumerate(zip(axes, data)):

        box_borders = [[0, 0, zmin], [exp_data['lengthX'], exp_data['lengthY'], zmax]]

        # Plot data
        row_axes[0].imshow(exp_data['data'][:, :, 0].T, origin='lower', cmap='gray')
        row_axes[1].imshow(exp_data['data'][:, :, -1].T, origin='lower', cmap='gray')
        plot_graph(row_axes[2], pred_mol, box_borders, class_colors, scatter_size, zmin, zmax)
        row_axes[3].imshow(sim[:, :, 0].T, origin='lower', cmap='gray')
        row_axes[4].imshow(sim[:, :, -1].T, origin='lower', cmap='gray')

        # Set labels
        row_axes[0].text(-0.08, 0.5, params[i]['label'], transform=row_axes[0].transAxes,
            fontsize=fontsize, va='center', ha='center', rotation='vertical')
        if i == 0:
            y = 1.08
            row_axes[0].text(0.5, y, 'Exp.\ AFM (far)', transform=row_axes[0].transAxes,
                fontsize=fontsize, va='center', ha='center')
            row_axes[1].text(0.5, y, 'Exp.\ AFM (close)', transform=row_axes[1].transAxes,
                fontsize=fontsize, va='center', ha='center')
            row_axes[2].text(0.5, y, 'Pred.\ geom.', transform=row_axes[2].transAxes,
                fontsize=fontsize, va='center', ha='center')
            row_axes[3].text(0.5, y, 'Sim.\ AFM (far)', transform=row_axes[3].transAxes,
                fontsize=fontsize, va='center', ha='center')
            row_axes[4].text(0.5, y, 'Sim.\ AFM (close)', transform=row_axes[4].transAxes,
                fontsize=fontsize, va='center', ha='center')

    plt.savefig(f'sims_exp.png', dpi=400)
