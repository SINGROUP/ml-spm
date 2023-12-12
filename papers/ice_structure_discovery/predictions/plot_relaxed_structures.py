#!/usr/bin/env python3

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mlspm.graph import MoleculeGraph
from mlspm.datasets import download_dataset

from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.field import HartreePotential
from ppafm.ocl.oclUtils import init_env

# Set matplotlib font rendering to use LaTex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

MM_TO_INCH = 1 / 25.4

def make_xyzs_xy_periodic(xyzs, Zs, lvec):
    xyzs_new = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            d = ix * lvec[1] + iy * lvec[2]
            xyzs_new.append(xyzs - d)
    xyzs_new = np.concatenate(xyzs_new, axis=0)
    Zs_new = np.concatenate([Zs]*9, axis=0)
    return xyzs_new, Zs_new

def get_sim(pot, mol, params):

    amp  = params['amp']
    dist = params['dist']
    nz   = params['nz']
    a    = params['rot_angle'] / 180 * np.pi

    xyzs = mol.array(xyz=True)
    Zs = mol.array(element=True).astype(np.int32)[:, 0]
    qs = pot

    rot = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [        0,          0, 1]
    ])
    rot_center = xyzs.mean(axis=0)
    xyzs_rot = np.dot(xyzs - rot_center, rot.T) + rot_center
    atoms_rot = np.concatenate([xyzs_rot, Zs[:, None]], axis=1)
    mol_rot = MoleculeGraph(atoms_rot, [], class_weights=mol.array(class_weights=True))

    h2o_xyzs = xyzs_rot[(Zs != 79) * (Zs != 29)]
    h2o_zs = h2o_xyzs[:, 2]
    h2o_xyzs = h2o_xyzs[h2o_zs > (h2o_zs.max() - 1.5)]

    df_steps = round(amp / 0.1)
    scan_size = np.array(params['scan_size'])
    offset = np.array(params['offset'])
    scan_start = (h2o_xyzs[:, :2].max(axis=0) + h2o_xyzs[:, :2].min(axis=0)) / 2 - scan_size / 2 + offset
    scan_end   = (h2o_xyzs[:, :2].max(axis=0) + h2o_xyzs[:, :2].min(axis=0)) / 2 + scan_size / 2 + offset
    scan_dim = params['scan_dim'] + (df_steps + nz - 1,)
    zmin = h2o_xyzs[:, 2].max() + dist
    zmax = zmin + (amp - 0.1) + (nz - 1) * 0.1
    scan_window = ((scan_start[0], scan_start[1], zmin), (scan_end[0], scan_end[1], zmax))

    afmulator = AFMulator(
        pixPerAngstrome=10,
        scan_dim=scan_dim,
        scan_window=scan_window,
        iZPP=8,
        df_steps=df_steps,
        rho={'dz2': -0.1}
    )

    X = afmulator(xyzs, Zs, qs, rot=rot, rot_center=rot_center)

    return X, scan_window, mol_rot

def crop_data(mol, sim, exp, sw):

    # Make a square box around the water molecules
    elems = mol.array(element=True)[:, 0]
    mask = (elems != 29) * (elems != 79)
    xyzs = mol.array(xyz=True)[mask]
    zs = xyzs[:, 2]
    mask2 = zs > (zs.max() - 1.5)
    xyzs = xyzs[mask2]
    xyzs_min = xyzs.min(axis=0)[:2]
    xyzs_max = xyzs.max(axis=0)[:2]
    xyzs_middle = (xyzs_min + xyzs_max) / 2
    xy_size = xyzs_max - xyzs_min
    if xy_size[0] > xy_size[1]:
        xyzs_min[1] = xyzs_middle[1] - xy_size[0] / 2
        xyzs_max[1] = xyzs_middle[1] + xy_size[0] / 2
    else:
        xyzs_min[0] = xyzs_middle[0] - xy_size[1] / 2
        xyzs_max[0] = xyzs_middle[0] + xy_size[1] / 2
    pad = min(
        6,
        xyzs_min[0] - sw[0][0],
        xyzs_min[1] - sw[0][1],
        sw[1][0] - xyzs_max[0],
        sw[1][1] - xyzs_max[1]
    )
    xyzs_min -= pad
    xyzs_max += pad

    x_res = (sw[1][0] - sw[0][0]) / sim.shape[0]
    y_res = (sw[1][1] - sw[0][1]) / sim.shape[1]

    x_start = round((xyzs_min[0] - sw[0][0]) / x_res)
    y_start = round((xyzs_min[1] - sw[0][1]) / y_res)
    x_end   = round((xyzs_max[0] - sw[0][0]) / x_res)
    y_end   = round((xyzs_max[1] - sw[0][1]) / y_res)

    sw = (
        (xyzs_min[0], xyzs_min[1], sw[0][2]),
        (xyzs_max[0], xyzs_max[1], sw[1][2])
    )
    sim = sim[x_start:x_end, y_start:y_end]
    exp = exp[x_start:x_end, y_start:y_end]

    return mol, sim, exp, sw

def get_data(params, exp_data_dir, sim_data_dir, classes):
    print('Loading', params['label'])

    exp_data = np.load(exp_data_dir / f'{params["exp_name"]}.npz')
    exp = exp_data['data'][..., -15:]
    params['scan_size'] = (exp_data['lengthX'], exp_data['lengthY'])
    params['scan_dim'] = exp.shape[:2]

    pot, xyzs, Zs = HartreePotential.from_file(str(sim_data_dir / f"{params['sim_name']}.xsf"), scale=-1)
    atoms = np.concatenate([xyzs, np.array(Zs)[:, None]], axis=1)
    mol = MoleculeGraph(atoms, [], classes=classes)

    sim, sw, mol = get_sim(pot, mol, params)
    mol, sim, exp, sw = crop_data(mol, sim, exp, sw)

    return mol, sim, exp, sw

def init_fig(num_rows, width=160, axes_gap=0.5, div_gap=5, left_margin=4, top_margin=4, num_cols=3, divisions=2, second_row_skip=3):

    effective_width = width - left_margin - divisions * num_cols * axes_gap - (divisions - 1) * div_gap
    ax_size = effective_width / (num_cols * divisions)

    left_margin *= MM_TO_INCH
    top_margin *= MM_TO_INCH
    axes_gap *= MM_TO_INCH
    ax_size *= MM_TO_INCH
    div_gap *= MM_TO_INCH
    width *= MM_TO_INCH
    second_row_skip *= MM_TO_INCH
    num_rows_div = math.ceil(num_rows / divisions)
    height = top_margin + second_row_skip + num_rows_div * (ax_size + axes_gap)
    fig = plt.figure(figsize=(width, height))
    print(width / MM_TO_INCH, height / MM_TO_INCH, ax_size / MM_TO_INCH)

    axes = []
    y = height - top_margin
    for i_row in range(num_rows_div):

        y -= ax_size
        for i_div in range(divisions):
            x = left_margin + i_div * (div_gap + num_cols * (ax_size + axes_gap))
            axes_ = []
            for _ in range(num_cols):
                rect = [x / width, y / height, ax_size / width, ax_size / height]
                ax = fig.add_axes(rect)
                ax.set_xticks([])
                ax.set_yticks([])
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(0.5)
                axes_.append(ax)
                x += (ax_size + axes_gap)
            axes.append(axes_)
            if len(axes) == num_rows:
                break
        
        if i_row == 0:
            y -= second_row_skip
            line_width = divisions * num_cols * (ax_size + axes_gap) + div_gap - axes_gap
            rect = [left_margin / width, y / height, line_width / width, (second_row_skip - axes_gap) / height]
            ax = fig.add_axes(rect)
            ax.hlines(y=0, xmin=-1, xmax=1, linestyles='dashed', colors='black', linewidth=0.8)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
        
        y -= axes_gap

    return fig, axes

def get_marker_size(z, zmin, zmax, scatter_size):
    return scatter_size * (z - zmin) / (zmax - zmin)

def plot_graph(ax, mol, box_borders, zmin, zmax, scatter_size, class_colors):

    Zs = mol.array(element=True)[:, 0]
    h2o_mask = (Zs != 29) * (Zs != 79)
    mol_pos = mol.array(xyz=True)
    mol_pos_h2o = mol_pos[h2o_mask]
    mol_pos_surface = mol_pos[~h2o_mask]
    mol_pos_h2o[:, 2] -= mol_pos_h2o[:, 2].max()

    s = get_marker_size(mol_pos_h2o[:, 2], zmin=zmin, zmax=zmax, scatter_size=scatter_size)
    s_surf = 0.2*get_marker_size(mol_pos_surface[:, 2], zmin=mol_pos_surface[:, 2].min() - 1.5, zmax=mol_pos_surface[:, 2].max(), scatter_size=scatter_size)
    if (s < 0).any():
        print(mol_pos_h2o[:, 2].min())
        raise ValueError('Encountered atom z position(s) below box borders.')
    
    c = [class_colors[atom.class_index] for atom in mol.atoms if atom.element in [1, 8]]
    c_surf = 'y' if Zs[~h2o_mask][0] == 79 else 'peru'
    
    ax.scatter(mol_pos_surface[:,0], mol_pos_surface[:,1], c=c_surf, s=s_surf, edgecolors='k', zorder=1, linewidths=0.15)

    zorder = np.argsort(mol_pos_h2o[:, 2])
    c = np.array(c)
    ax.scatter(mol_pos_h2o[zorder, 0], mol_pos_h2o[zorder, 1], c=c[zorder], s=s[zorder], edgecolors='k', zorder=2, linewidths=0.15)

    for b in mol.bonds:
        pos = np.vstack([mol_pos_h2o[b[0]], mol_pos[b[1]]])
        ax.plot(pos[:,0], pos[:,1], 'k', linewidth=0.15, zorder=1)
    
    ax.set_xlim(box_borders[0][0], box_borders[1][0])
    ax.set_ylim(box_borders[0][1], box_borders[1][1])
    ax.set_aspect('equal', 'box')

if __name__ == '__main__':

    exp_data_dir = Path('./exp_data/')
    sim_data_dir = Path('./relaxed_structures/')
    scatter_size = 6
    zmin = -5.0
    zmax = 0.5
    classes = [[1], [8], [29, 79]]
    class_colors = ['w', 'r']
    fontsize = 7

    init_env(i_platform=1)

    params = [
        {'sim_name': 'hartree_A', 'exp_name': 'Ying_Jiang_7'  , 'label': 'A', 'dist': 5.00, 'rot_angle': -145.000, 'amp': 2.0, 'nz':  9, 'offset': ( 1.0,  0.0)},
        {'sim_name': 'hartree_B', 'exp_name': 'Chen_CO'       , 'label': 'B', 'dist': 5.00, 'rot_angle':   80.000, 'amp': 2.0, 'nz': 15, 'offset': (-1.0,  0.5)},
        {'sim_name': 'hartree_C', 'exp_name': 'Ying_Jiang_1'  , 'label': 'C', 'dist': 4.50, 'rot_angle':  -95.000, 'amp': 2.0, 'nz': 12, 'offset': ( 0.0, -1.5)},
        {'sim_name': 'hartree_D', 'exp_name': 'Ying_Jiang_2_1', 'label': 'D', 'dist': 4.50, 'rot_angle':  180.000, 'amp': 2.0, 'nz':  9, 'offset': ( 1.0,  0.5)},
        {'sim_name': 'hartree_E', 'exp_name': 'Ying_Jiang_2_2', 'label': 'E', 'dist': 4.50, 'rot_angle':  -60.000, 'amp': 2.0, 'nz':  9, 'offset': ( 0.0,  0.0)},
        {'sim_name': 'hartree_F', 'exp_name': 'Ying_Jiang_3'  , 'label': 'F', 'dist': 4.50, 'rot_angle': -145.000, 'amp': 2.0, 'nz':  9, 'offset': ( 0.0, -3.5)},
        {'sim_name': 'hartree_G', 'exp_name': 'Ying_Jiang_5'  , 'label': 'G', 'dist': 4.50, 'rot_angle':   94.000, 'amp': 2.0, 'nz': 11, 'offset': ( 2.0,  0.0)},
        {'sim_name': 'hartree_H', 'exp_name': 'Ying_Jiang_6'  , 'label': 'H', 'dist': 4.30, 'rot_angle': -165.974, 'amp': 2.0, 'nz': 10, 'offset': ( 1.5,  2.0)},
        # {'sim_name': 'hartree_I', 'exp_name': 'Ying_Jiang_4'  , 'label': 'I', 'dist': 4.80, 'rot_angle':  -25.000, 'amp': 2.0, 'nz':  7, 'offset': ( 8.0,  3.5)} # Takes a lot of video memory
    ]

    # Download datasets
    download_dataset("AFM-ice-exp", exp_data_dir)
    download_dataset("AFM-ice-relaxed", sim_data_dir)

    data = [get_data(p, exp_data_dir, sim_data_dir, classes) for p in params]
    fig, axes = init_fig(num_rows=len(data), num_cols=3, divisions=2)

    for i, (row_axes, (mol, sim, exp, sw)) in enumerate(zip(axes, data)):

        box_borders = [[sw[0][0], sw[0][1], zmin], [sw[1][0], sw[1][1], zmax]]

        # Plot data
        plot_graph(row_axes[0], mol, box_borders, zmin=zmin, zmax=zmax, scatter_size=scatter_size, class_colors=class_colors)
        row_axes[1].imshow(sim[:, :, -1].T, origin='lower', cmap='gray')
        row_axes[2].imshow(exp[:, :, -1].T, origin='lower', cmap='gray')

        # Set labels
        row_axes[0].text(-0.08, 0.5, params[i]['label'], transform=row_axes[0].transAxes,
            fontsize=fontsize, va='center', ha='center', rotation='vertical')
        if i < 2:
            y = 1.08
            row_axes[0].text(0.5, y, 'Opt.\ geom.', transform=row_axes[0].transAxes,
                fontsize=fontsize, va='center', ha='center')
            row_axes[1].text(0.5, y, 'Sim.\ AFM', transform=row_axes[1].transAxes,
                fontsize=fontsize, va='center', ha='center')
            row_axes[2].text(0.5, y, 'Exp.\ AFM', transform=row_axes[2].transAxes,
                fontsize=fontsize, va='center', ha='center')

    plt.savefig(f'sims_relaxed.png', dpi=400)
