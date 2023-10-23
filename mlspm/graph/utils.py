from typing import Tuple

import numpy as np
import torch
from scipy.stats import multivariate_normal
from skimage import feature, measure

from .._c.bindings import match_template_pool


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
