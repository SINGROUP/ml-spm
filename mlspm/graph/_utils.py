import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from scipy.stats import multivariate_normal
from skimage import feature, measure

from .._c.bindings import match_template_pool, peak_dist
from ..utils import write_to_xyz
from ._molecule_graph import MoleculeGraph

VDW_RADII = {1: 1.487, 6: 1.908, 7: 1.78, 8: 1.661, 9: 1.75, 14: 1.9, 15: 2.1, 16: 2.0, 17: 1.948, 35: 2.22}

# Reference: http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
# fmt:off
BOND_LENGTHS = {
     1: { 1: 0.74,  6: 1.09,  7: 1.01,  8: 0.96,  9: 0.92, 14: 1.48, 15: 1.42, 16: 1.34, 17: 1.27, 35: 1.41, 53: 1.61},
     6: {           6: 1.54,  7: 1.47,  8: 1.43,  9: 1.33, 14: 1.86, 15: 1.87, 16: 1.81, 17: 1.77, 35: 1.94, 53: 2.13},
     7: {                     7: 1.46,  8: 1.44,  9: 1.39, 14: 1.72, 15: 1.77, 16: 1.68, 17: 1.91, 35: 2.14, 53: 2.22},
     8: {                               8: 1.48,  9: 1.42, 14: 1.61, 15: 1.60, 16: 1.51, 17: 1.64, 35: 1.72, 53: 1.94},
     9: {                                         9: 1.43, 14: 1.56, 15: 1.56, 16: 1.58, 17: 1.66, 35: 1.78, 53: 1.87},
    14: {                                                  14: 2.34, 15: 2.27, 16: 2.10, 17: 2.04, 35: 2.16, 53: 2.40},
    15: {                                                            15: 2.21, 16: 2.10, 17: 2.04, 35: 2.22, 53: 2.43},
    16: {                                                                      16: 2.04, 17: 2.01, 35: 2.25, 53: 2.34},
    17: {                                                                                17: 1.99, 35: 2.18, 53: 2.43},
    35: {                                                                                          35: 2.28, 53: 2.48},
    53: {                                                                                                    53: 2.66}
}
# fmt: on


def _find_peaks_cpu(
    pos_dist: np.ndarray, box_borders: np.ndarray, match_threshold: float, std: float, method: str
) -> Tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    n_xyz = pos_dist.shape[1:]
    res = [(box_borders[1][i] - box_borders[0][i]) / (n_xyz[i] - 1) for i in range(3)]
    pos_dist[pos_dist < 1e-4] = 0  # Very small values cause instabilities in ZNCC values

    # Create reference gaussian peak to compare against
    r = 3 * std + 1e-6
    r = [r - (r % res[i]) for i in range(3)]
    x_ref, y_ref, z_ref = [np.arange(-r[i], r[i] + 1e-6, res[i]) for i in range(3)]
    ref_grid = np.stack(np.meshgrid(x_ref, y_ref, z_ref, indexing="ij"), axis=-1)
    ref_peak = multivariate_normal.pdf(ref_grid, mean=[0, 0, 0], cov=std**2)

    # Match the reference gaussian peak shape with the position distributions
    if method in ["mad", "msd", "mad_norm", "msd_norm"]:
        matches = match_template_pool(pos_dist, ref_peak, method=method)
    else:
        matches = []
        for d in pos_dist:
            matches.append(feature.match_template(d, ref_peak, pad_input=True, mode="constant", constant_values=0))
        matches = np.stack(matches, axis=0)

    # Threshold the match map
    if method == "zncc":
        threshold_masks = matches > match_threshold
    else:
        threshold_masks = matches < match_threshold

    # Loop over batch items to label matches and find atom positions
    xyzs = []
    labels = []
    for match, threshold_mask in zip(matches, threshold_masks):
        # Label connected regions
        labels_, num_atoms = measure.label(threshold_mask, return_num=True)

        # Loop over labelled regions to find atom positions
        xyzs_ = []
        for target_label in range(1, num_atoms + 1):
            # Find best matching xyz position from the labelled region
            match_masked = np.ma.array(match, mask=(labels_ != target_label))
            best_ind = match_masked.argmax() if method == "zncc" else match_masked.argmin()
            best_ind = np.unravel_index(best_ind, match_masked.shape)
            xyz = [box_borders[0][i] + res[i] * best_ind[i] for i in range(3)]

            xyzs_.append(xyz)

        xyzs.append(np.array(xyzs_))
        labels.append(labels_)

    labels = np.stack(labels, axis=0)

    return xyzs, matches, labels


def _find_peaks_cuda(
    pos_dist: torch.Tensor, box_borders: np.ndarray, match_threshold: float, std: float, method: str
) -> Tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    # This import triggers a JIT compilation, so let's put it here inside the function, so that you
    # don't waste time on compilation when you don't need it.
    from .._cuda.bindings import ccl, find_label_min
    from .._cuda.bindings import match_template as match_template_cuda

    if method == "zncc":
        raise NotImplementedError("zncc not implemented for cuda tensors.")

    n_xyz = pos_dist.shape[1:]
    res = [(box_borders[1][i] - box_borders[0][i]) / (n_xyz[i] - 1) for i in range(3)]

    # Create reference gaussian peak to compare against
    r = 3 * std + 1e-6
    r = [r - (r % res[i]) for i in range(3)]
    x_ref, y_ref, z_ref = [np.arange(-r[i], r[i] + 1e-6, res[i]) for i in range(3)]
    ref_grid = np.stack(np.meshgrid(x_ref, y_ref, z_ref, indexing="ij"), axis=-1)
    ref_peak = multivariate_normal.pdf(ref_grid, mean=[0, 0, 0], cov=std**2)
    ref_peak = torch.from_numpy(ref_peak).to(pos_dist.device).float()

    # Match the reference gaussian peak shape with the position distributions
    matches = match_template_cuda(pos_dist, ref_peak, method=method)

    # Threshold the match map
    threshold_masks = matches < match_threshold

    # Label matched regions
    labels = ccl(threshold_masks)

    # Find minimum indices in labelled regions
    min_inds = find_label_min(matches, labels)

    # Convert indices into real-space coordinates
    xyz_start = torch.tensor(box_borders[0], device=pos_dist.device, dtype=pos_dist.dtype)
    res = torch.tensor(res, device=pos_dist.device, dtype=pos_dist.dtype)
    xyzs = [xyz_start + res * m.type(pos_dist.dtype) for m in min_inds]

    return xyzs, matches, labels


def find_gaussian_peaks(
    pos_dist: np.ndarray | torch.Tensor, box_borders: np.ndarray, match_threshold: float = 0.7, std: float = 0.3, method: str = "mad"
) -> Tuple[list[np.ndarray], np.ndarray, np.ndarray] | Tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Find real-space positions of gaussian peaks in a 3D position distribution grid.

    Arguments:
        pos_dist: Position distribution array. Should be of shape (n_batch, nx, ny, nz).
        box_borders: Real-space extent of the distribution grid in angstroms. The array should be of the form
            ((x_start, y_start, z_start), (x_end, y_end, z_end)).
        match_threshold: Detection threshold for matching. Regions above the threshold are chosen for method 'zncc',
            and regions below the threshold are chosen for methods 'mad', 'msd, 'mad_norm', and 'msd_norm'.
        std: Standard deviation of peaks to search for in angstroms.
        method: 'zncc', 'mad', 'msd', 'mad_norm', or 'msd_norm. Matching method to use. Either zero-normalized
            cross correlation ('zncc'), mean absolute distance ('mad'), mean squared distance ('msd'), or the
            normalized version of the latter two ('mad_norm', 'msd_norm').

    Returns: xyzs, match, labels
        xyzs: list of np.ndarray or torch.Tensor of shape (num_atoms, 3). Positions of the found atoms.
            Each item in the list corresponds one batch item.
        matches: np.ndarray or torch.Tensor of same shape as input pos_dist. Array of matching values.
            For method 'zncc' larger values, and for 'mad', 'msd', 'mad_norm', and 'msd_norm' smaller
            values correspond to better match.
        labels: np.ndarray or torch.Tensor of same shape as input pos_dist. Labelled regions where
            match is better than match_threshold.
    """

    if method not in ["zncc", "mad", "msd", "mad_norm", "msd_norm"]:
        raise ValueError(f"Unknown matching method `{method}`.")

    if isinstance(pos_dist, torch.Tensor):
        if pos_dist.device == torch.device("cpu"):
            xyzs, matches, labels = _find_peaks_cpu(pos_dist.numpy(), box_borders, match_threshold, std, method)
            xyzs = [torch.from_numpy(xyz).type(pos_dist.dtype) for xyz in xyzs]
            matches = torch.from_numpy(matches).type(pos_dist.dtype)
            labels = torch.from_numpy(labels).type(pos_dist.dtype)
        else:
            xyzs, matches, labels = _find_peaks_cuda(pos_dist, box_borders, match_threshold, std, method)
    else:
        xyzs, matches, labels = _find_peaks_cpu(pos_dist, box_borders, match_threshold, std, method)

    return xyzs, matches, labels


def make_position_distribution(mols, box_borders, box_res=(0.125, 0.125, 0.1), std=0.3):
    """
    Make a distribution on a grid based on atom positions. Each atom is represented by
    a normal distribution.

    Arguments:
        mols: list of MoleculeGraph. List of molecules with atom positions.
        box_borders: tuple ((x_start, y_start, z_start),(x_end, y_end, z_end)). Real-space extent of the grid
            in angstroms.
        box_res: tuple (x_res, y_res, z_res). Real-space size of each voxel in angstroms.
        std: float. Standard deviation of normal distribution for each atom in angstroms.

    Returns: np.ndarray of size (n_batch, n_x, n_y, n_z).
    """
    n_xyz = [int((box_borders[1][i] - box_borders[0][i]) / box_res[i] + 1.01) for i in range(3)]
    atoms = [m.array(xyz=True) if len(m) > 0 else np.empty((0, 3)) for m in mols]
    xyz_start = box_borders[0]
    pos_dist = peak_dist(atoms, n_xyz, xyz_start, box_res, std)
    return pos_dist


def shift_mols_window(molecules, scan_windows, start=(0, 0)):
    """
    Shift molecule xy coordinates to use the same scan window. All molecules should have the same
    scan window size.

    Arguments:
        molecules: list of MoleculeGraph. Molecule whose atom positions to shift.
        scan_windows: list of np.ndarray of shape (n_mol, 2, 3). Scan window for each molecule.
        start: tuple of floats (x, y). Start of the new scan window.

    Returns:
        new_molecules: list of MoleculeGraph. Molecule with shifted atom coordinates.
        new_scan_window: tuple ((x_start, y_start), (x_end, y_end)). New scan window.
    """

    assert len(molecules) == len(scan_windows)

    swx = scan_windows[:, 1, 0] - scan_windows[:, 0, 0]
    swy = scan_windows[:, 1, 1] - scan_windows[:, 0, 1]
    if not (np.allclose(swx, swx[0]) and np.allclose(swy, swy[0])):
        raise ValueError("All molecules do not have the same scan window size.")

    x_size, y_size = swx[0], swy[0]
    new_scan_window = (start, (start[0] + x_size, start[1] + y_size))

    new_molecules = []
    for mol, sw in zip(molecules, scan_windows):
        shift = (start[0] - sw[0][0], start[1] - sw[0][1])
        new_molecules.append(mol.transform_xy(shift=shift))

    return new_molecules, new_scan_window


def add_rotation_reflection_graph(X, mols, box_borders, num_rotations=1, reflections=True, crop=None, per_batch_item=True):
    """
    Random rotation and reflection of AFM images and corresponding molecule graphs.

    Arguments:
        X: np.ndarray of shape (batch, x, y, z). AFM images.
        mols: list of MoleculeGraph. Molecule graphs corresponding to the AFM images.
        box_borders: tuple ((x_start, y_start, z_start), (x_end, y_end, z_end)). Real-space extent of the
            AFM region in angstroms.
        num_rotations: int. How many rotations for each batch item.
        reflections: bool. Whether to augment with reflections.
        crop: None or tuple (x_size, y_size) or 'max'. If tuple, then output batch is cropped to specified
            size. If 'max, the crop region will be maximized to fit into the rotated image. Atoms outside the
            cropped region in the molecule graphs are deleted. The crop region is centered to the middle of
            the image.
        per_batch_item: bool. If True, rotation is randomized per batch item, otherwise same rotation for all.

    Returns:
        X: np.ndarray of shape (batch*num_rotations, x_new, y_new, z). Rotation-augmented AFM images.
        mols: list of MoleculeGraph. New rotated molecule graphs.
    """

    box_center = ((box_borders[1][0] + box_borders[0][0]) / 2, (box_borders[1][1] + box_borders[0][1]) / 2)

    mols_aug = []
    X_aug = [[] for _ in range(len(X))]
    for _ in range(num_rotations):
        if per_batch_item:
            rotations = 360 * np.random.rand(len(X[0]))
            flip = [np.random.randint(2) == 1 if reflections else False for _ in range(len(X[0]))]
        else:
            rotations = np.array([360 * np.random.rand()] * len(X[0]))
            flip = [np.random.randint(2) == 1 if reflections else False] * len(X[0])

        for k, x in enumerate(X):
            x = x.copy()
            for i in range(x.shape[0]):
                for j in range(x.shape[-1]):
                    x[i, :, :, j] = np.array(Image.fromarray(x[i, :, :, j]).rotate(rotations[i], resample=Image.BICUBIC))
                if flip[i]:
                    x[i] = x[i, :, ::-1]
            X_aug[k].append(x)

        for i, mol in enumerate(mols):
            mols_aug.append(mol.transform_xy(rot_xy=rotations[i], flip_y=flip[i], center=box_center))

    X_aug = [np.concatenate(x, axis=0) for x in X_aug]

    if crop is not None:
        if crop == "max":
            a = (rotations % 90) / 180 * np.pi
            max_dist = int((X[0].shape[1] / (np.cos(a) + np.sin(a))).min())
            crop = (max_dist, max_dist)

        x_start = (X_aug[0].shape[1] - crop[0]) // 2
        y_start = (X_aug[0].shape[2] - crop[1]) // 2

        X_aug, mols_aug, box_borders = crop_graph(X_aug, mols_aug, (x_start, y_start), crop, box_borders)

    return X_aug, mols_aug, box_borders


def find_bonds(molecules: list[np.ndarray], tolerance=0.2) -> list[list[Tuple[int, int]]]:
    """
    Find bonds in molecules based on atomic distances and a tabulated bond lengths.

    Arguments:
        molecules: Molecule atom position and elements. list of arrays of shape (num_atoms, 4), where each row corresponds
            to one atom with [x, y, z, element].
        tolerance: float. Two atoms are bonded if their distance is at most by a factor of 1+tolerance as long as the table
            value for the bond length.

    Returns: Indices of bonded atoms for each molecule.
    """
    bonds = []
    for mol in molecules:
        bond_ind = []
        for i in range(len(mol)):
            for j in range(len(mol)):
                if j <= i:
                    continue
                atom_i = mol[i]
                atom_j = mol[j]
                r = np.linalg.norm(atom_i[:3] - atom_j[:3])
                elems = sorted([atom_i[-1], atom_j[-1]])
                bond_length = BOND_LENGTHS[elems[0]][elems[1]]
                if r < (1 + tolerance) * bond_length:
                    bond_ind.append((i, j))
        bonds.append(bond_ind)
    return bonds


def threshold_atoms_bonds(molecules, threshold=-1.0, use_vdW=False):
    """
    Remove atoms and corresponding bonds beyond threshold depth in molecules.
    Arguments:
        molecules: list of MoleculeGraph. Molecules to threshold.
        threshold: float. Deepest z-coordinate for included atoms (top is 0).
        use_vdW: Boolean. Whether to add vdW radii to the atom z coordinates when calculating depth.
    Returns:
        new_molecules: list of MoleculeGraph. Molecules with deep atoms removed.
    """
    new_molecules = []
    for mol in molecules:
        if len(mol) == 0:
            new_molecules.append(mol.copy())
            continue
        zs = mol.array(xyz=True)[:, 2].copy()
        if use_vdW:
            zs += np.fromiter(map(lambda i: VDW_RADII[i], mol.array(element=True)[:, 0]), dtype=np.float64)
        zs -= zs.max()
        remove_inds = np.where(zs < threshold)[0]
        new_molecule, removed = mol.remove_atoms(remove_inds)
        new_molecules.append(new_molecule)
    return new_molecules


def crop_graph(X, mols, start, size, box_borders, new_start=(0.0, 0.0)):
    """
    Crop AFM images and molecule graphs in a batch to a different size.
    Arguments:
        X: list of np.ndarray of shape (batch, x, y, z). AFM images.
        mols: list of MoleculeGraph. Molecule graphs corresponding to the AFM images.
        start: tuple of ints (x, y). Start pixels for crop in x and y directions.
        size: tuple of ints (x, y). Size of cropped region in x and y directions.
        box_borders: tuple ((x_start, y_start, z_start), (x_end, y_end, z_end)). Real-space extent of the
            AFM region in angstroms.
        new_start: tuple of ints (x, y). The start coordinates of the cropped region in angstroms.
    Returns:
        X: list of np.ndarray of shape (batch, size[0], size[1], z). Cropped AFM images.
        mols: list of MoleculeGraph. Cropped molecule graphs.
        box_borders_cropped: tuple ((x_start, y_start, z_start), (x_end, y_end, z_end)). Real-space extent
            of the cropped region.
    """

    x_size, y_size = X[0].shape[1], X[0].shape[2]
    x_start, y_start = start
    width, height = size

    x_res = (box_borders[1][0] - box_borders[0][0]) / (x_size - 1)
    y_res = (box_borders[1][1] - box_borders[0][1]) / (y_size - 1)
    x_shift = new_start[0] - (box_borders[0][0] + x_start * x_res)
    y_shift = new_start[1] - (box_borders[0][1] + y_start * y_res)
    box_borders_cropped = (
        (new_start[0], new_start[1], box_borders[0][2]),
        (new_start[0] + (width - 1) * x_res, new_start[0] + (height - 1) * y_res, box_borders[1][2]),
    )

    X = [x[:, x_start : x_start + width, y_start : y_start + height] for x in X]
    mols = [m.transform_xy(shift=(x_shift, y_shift)).crop_atoms(box_borders_cropped) for m in mols]

    return X, mols, box_borders_cropped


def save_graphs_to_xyzs(
    molecules: list[MoleculeGraph],
    classes: list[list[int | str]],
    outfile_format: str = "./{ind}_graph.xyz",
    start_ind: int = 0,
    verbose: int = 1,
):
    """
    Save molecule graphs to xyz files.

    Arguments:
        molecules: Molecule graphs to save.
        classes: Chemical elements for atom classification. Either atomic numbers of chemical symbols.
        outfile_format: Formatting string for saved files. Sample index is available in variable "ind".
        start_ind: Index where file numbering starts.
        verbose: Whether to print output information.
    """

    ind = start_ind
    for mol in molecules:
        outfile = outfile_format.format(ind=ind)
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if len(mol) > 0:
            mol_xyz = mol.array(xyz=True)
            mol_elements = np.array([classes[int(m)][0] for m in mol.array(class_index=True).squeeze(1)])[:, None]
            mol_arr = np.append(mol_xyz, mol_elements, axis=1)
        else:
            mol_arr = np.empty((0, 4))

        write_to_xyz(mol_arr, outfile, verbose=verbose)

        ind += 1


def make_box_borders(shape: Tuple[int, int], res: Tuple[float, float], z_range: Tuple[float, float]) -> np.ndarray:
    """
    Make grid box borders for a given grid xy shape.

    Arguments:
        shape: Grid xy shape.
        res: Grid xy pixel resolution in Ånströms.
        z_range: Grid z start and end coordinates in Ånströms.

    Returns: Box start and end coordinates in the form ((x_start, y_start, z_start), (x_end, y_end, z_end)).
    """
    # fmt:off
    box_borders = np.array([
        [                      0,                       0, z_range[0]],
        [res[0] * (shape[0] - 1), res[1] * (shape[1] - 1), z_range[1]]
    ])  # fmt:on
    return box_borders
