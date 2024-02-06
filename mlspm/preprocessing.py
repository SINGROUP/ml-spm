import random
from typing import List, Literal, Optional, Tuple

import numpy as np
import scipy.ndimage as nimg
from PIL import Image


def add_norm(Xs: list[np.ndarray], per_layer: bool = True):
    """
    Normalize arrays by subracting the mean and dividing by standard deviation. In-place operation.

    Arguments:
        Xs: Arrays to normalize. Each array should be of shape ``(batch_size, ...)``.
        per_layer: If True, normalized separately for each element in last axis of **Xs**.
    """
    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            if per_layer:
                for i in range(sh[-1]):
                    X[j, ..., i] = (X[j, ..., i] - np.mean(X[j, ..., i])) / np.std(X[j, ..., i])
            else:
                X[j] = (X[j] - np.mean(X[j])) / np.std(X[j])


def add_noise(Xs: list[np.ndarray], c: float = 0.1, randomize_amplitude: bool = False, normal_amplitude: bool = False):
    """
    Add uniform random noise to arrays. In-place operation.

    Arguments:
        Xs: Arrays to add noise to. Each array should be of shape ``(batch_size, ...)``.
        c: Amplitude of noise. Is multiplied by (max-min) of sample.
        randomize_amplitude: If True, noise amplitude is uniform random in ``[0, c]`` for each sample in the batch.
        normal_amplitude: If True and ``randomize_amplitude==True``, then instead of uniform, the noise amplitude is distributed
            like the absolute value of a normally distributed variable with zero mean and standard deviation equal to **c**.
    """
    for X in Xs:
        sh = X.shape
        R = np.random.rand(*sh) - 0.5
        if randomize_amplitude:
            if normal_amplitude:
                amp = np.abs(np.random.normal(0, c, sh[0]))
            else:
                amp = np.random.uniform(0.0, 1.0, sh[0]) * c
        else:
            amp = [c] * sh[0]
        for j in range(sh[0]):
            X[j] += R[j] * amp[j] * (X[j].max() - X[j].min())


def add_cutout(Xs: list[np.ndarray], n_holes: int = 5):
    """
    Randomly add cutouts (rectangular patches of zeros) to images. In-place operation.

    Arguments:
        Xs: Arrays to add cutouts to. Each array should be of shape ``(batch_size, x, y, z)``.
        n_holes: Maximum number of cutouts to add.
    """

    def get_random_eraser(input_img, p=0.2, s_l=0.001, s_h=0.01, r_1=0.1, r_2=1.0 / 0.1):
        """
        p : the probability that random erasing is performed
        s_l, s_h : minimum / maximum proportion of erased area against input image
        r_1, r_2 : minimum / maximum aspect ratio of erased area
        """

        sh = input_img.shape
        img_h, img_w = [sh[0], sh[1]]

        if np.random.uniform(0, 1) > p:
            return input_img

        while True:
            s = np.exp(np.random.uniform(np.log(s_l), np.log(s_h))) * img_h * img_w
            r = np.exp(np.random.uniform(np.log(r_1), np.log(r_2)))

            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        input_img[top : top + h, left : left + w] = 0.0

        return input_img

    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            for i in range(sh[3]):
                for _ in range(n_holes):
                    X[j, :, :, i] = get_random_eraser(X[j, :, :, i])


def add_gradient(Xs: list[np.ndarray], c: float = 0.3):
    """
    Add a constant gradient plane with random direction to arrays. In-place operation.

    Arguments:
        Xs: Arrays to add gradients to. Each array should be of shape ``(batch_size, x, y, z)``.
        c: Maximum range of gradient plane as a fraction of the range of the array values.
    """
    assert len(set([X.shape for X in Xs])) == 1  # All same shape
    x, y = np.meshgrid(np.arange(0, Xs[0].shape[1]), np.arange(0, Xs[0].shape[2]), indexing="ij")
    for i in range(Xs[0].shape[0]):
        c_eff = c * np.random.rand()
        angle = 2 * np.pi * np.random.rand()
        n = [np.cos(angle), np.sin(angle), 1]
        z = -(n[0] * x + n[1] * y)
        z -= z.mean()
        z /= np.ptp(z)
        for X in Xs:
            X[i] += z[:, :, None] * c_eff * np.ptp(X[i])


def rand_shift_xy_trend(Xs: list[np.ndarray], max_layer_shift: float = 0.02, max_total_shift: float = 0.1):
    """
    Randomly shift z-layers in the xy-plane. Each shift is relative to previous one. In-place operation.

    Arguments:
        Xs: Arrays to shift. Each array should be of shape ``(batch_size, x, y, z)``.
        shift_step_max: Maximum fraction of image size by which to shift for each layer. Should be in the interval ``[0, 1]``.
        max_shift_total: Maximum fraction of image size by which to shift in total. Should be in the interval ``[0, 1]`` and
            more than **shift_step_max**.
    """

    if not (0 <= max_layer_shift <= 1.0):
        raise ValueError(f"Max layer shift should be in the interval from 0 to 1, but got {max_layer_shift}")

    if not (0 <= max_total_shift <= 1.0):
        raise ValueError(f"Max total shift should be in the interval from 0 to 1, but got {max_total_shift}")

    if max_layer_shift > max_total_shift:
        raise ValueError("Max layer shift cannot be larger than the max total shift.")

    for X in Xs:
        sh = X.shape

        # Calculate max possible shifts in pixels between neighbor slices and maximum shift
        xy_max = np.maximum(sh[1], sh[2])
        max_slice_shift_pix = np.floor(xy_max * max_layer_shift).astype(int)
        max_trend_pix = np.floor(xy_max * max_total_shift).astype(int)

        for j in range(sh[0]):  # Iterate over batch items
            # Calculate values of random shift for slices in reverse order
            # 0 value for closest slice and biggest values for further slices
            rand_shift = np.zeros((sh[3], 2))
            for i in range(sh[3] - 1, 0, -1):
                shift_values = [random.choice(np.arange(-max_slice_shift_pix, max_slice_shift_pix + 1)) for _ in range(2)]
                for slice_ind in range(i):
                    rand_shift[slice_ind, :] = rand_shift[slice_ind, :] + shift_values
                rand_shift = np.clip(rand_shift, -max_trend_pix, max_trend_pix).astype(int)  # Clip too big shifts

            # Apply shifts
            for i, (shift_x, shift_y) in enumerate(rand_shift):
                X[j, :, :, i] = nimg.shift(X[j, :, :, i], (shift_y, shift_x), mode="mirror")


def top_atom_to_zero(xyzs: list[np.ndarray]):
    """
    Set the z coordinate of the highest atom in each molecule to 0. In-place operation.

    Arguments:
        xyzs: Molecule arrays to modify. Each array should be of shape ``(num_atoms, :)``, such that
            the first three elements on the second axis are the xyz coordinates.
    """
    new_xyzs = []
    for xyz in xyzs:
        xyz[:, 2] -= xyz[:, 2].max()
        new_xyzs.append(xyz)
    return new_xyzs


def interpolate_and_crop(
    Xs: list[np.ndarray], real_dim: Tuple[int, int], target_res: float = 0.125, target_multiple: int = 8
) -> list[np.ndarray]:
    """
    Interpolate a batch of AFM images to target resolution and crop to a target multiple of pixels in the xy plane.

    Arguments:
        X: AFM images to interpolate and crop. Each array should be of shape ``(batch_size, x, y, z)``.
        real_dim: Real-space size of AFM image region in x- and y-directions in Ångstroms.
        target_res: Target size for a pixel in angstroms.
        target_multiple: Target multiple of pixels of output image.

    Returns:
        Interpolated and cropped AFM images.
    """

    resized_shape = (int(real_dim[1] // target_res), int(real_dim[0] // target_res))
    for k, x in enumerate(Xs):
        # Interpolate for correct resolution
        Xs_ = np.empty((x.shape[0], resized_shape[1], resized_shape[0], x.shape[-1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[-1]):
                Xs_[i, :, :, j] = np.array(Image.fromarray(x[i, :, :, j]).resize(resized_shape, Image.BILINEAR))

        # Crop for correct multiple of pixels
        dx = resized_shape[1] % target_multiple
        x_start = int(np.floor(dx / 2))
        x_end = resized_shape[1] - int(np.ceil(dx / 2))

        dy = resized_shape[0] % target_multiple
        y_start = int(np.floor(dy / 2))
        y_end = resized_shape[0] - int(np.ceil(dy / 2))

        Xs[k] = Xs_[:, x_start:x_end, y_start:y_end]

    return Xs


def minimum_to_zero(Ys: List[np.ndarray]):
    """
    Shift values in arrays such that minimum is at zero. In-place operation.

    Arguments:
        Ys: Arrays of shape (batch_size, ...).
    """
    for Y in Ys:
        for j in range(Y.shape[0]):
            Y[j] -= Y[j].min()


def add_rotation_reflection(
    X: List[np.ndarray],
    Y: List[np.ndarray],
    reflections: bool = True,
    multiple: int = 2,
    crop: Optional[Tuple[int]] = None,
    per_batch_item: bool = False,
):
    """
    Augment batch with random rotations and reflections.

    Arguments:
        X: AFM images to augment. Each array should be of shape ``(batch_size, x, y, z)``.
        Y: Reference image descriptors to augment. Each array should be of shape ``(batch, x, y)``.
        reflections: Whether to augment with reflections. If True, each rotation is randomly reflected with 50% probability.
        multiple: Multiplier for how many rotations to generate for every sample.
        crop: If not None, then output batch is cropped to specified size ``(x_size, y_size)`` in the middle of the image.
        per_batch_item: If True, rotation is randomized per batch item, otherwise same rotation for all.

    Returns:
        Tuple (**X**, **Y**), where

        - **X** - Batch of rotation-augmented AFM images of shape ``(batch*multiple, x_new, y_new, z)``.
        - **Y** - Batch of rotation-augmented reference image descriptors of shape ``(batch*multiple, x_new, y_new)``
    """

    X_aug = [[] for _ in range(len(X))]
    Y_aug = [[] for _ in range(len(Y))]

    for _ in range(multiple):
        if per_batch_item:
            rotations = 360 * np.random.rand(len(X[0]))
        else:
            rotations = [360 * np.random.rand()] * len(X[0])
        if reflections:
            flip = np.random.randint(2)
        for k, x in enumerate(X):
            x = x.copy()
            for i in range(x.shape[0]):
                for j in range(x.shape[-1]):
                    x[i, :, :, j] = np.array(Image.fromarray(x[i, :, :, j]).rotate(rotations[i], resample=Image.BICUBIC))
            if flip:
                x = x[:, :, ::-1]
            X_aug[k].append(x)
        for k, y in enumerate(Y):
            y = y.copy()
            for i in range(y.shape[0]):
                y[i, :, :] = np.array(Image.fromarray(y[i, :, :]).rotate(rotations[i], resample=Image.BICUBIC))
            if flip:
                y = y[:, :, ::-1]
            Y_aug[k].append(y)

    X = [np.concatenate(x, axis=0) for x in X_aug]
    Y = [np.concatenate(y, axis=0) for y in Y_aug]

    if crop is not None:
        x_start = (X[0].shape[1] - crop[0]) // 2
        y_start = (X[0].shape[2] - crop[1]) // 2
        X = [x[:, x_start : x_start + crop[0], y_start : y_start + crop[1]] for x in X]
        Y = [y[:, x_start : x_start + crop[0], y_start : y_start + crop[1]] for y in Y]

    return X, Y


def random_crop(
    X: List[np.ndarray],
    Y: List[np.ndarray],
    min_crop: float = 0.5,
    max_aspect: float = 2.0,
    multiple: int = 8,
    distribution: Literal["flat", "exp-log"] = "flat",
):
    """
    Randomly crop images in a batch to a different size and aspect ratio.

    Arguments:
        X: AFM images to crop. Each array should be of shape ``(batch_size, x, y, z)``.
        Y: Reference image descriptors to crop. Each array should be of shape ``(batch, x, y)``.
        min_crop: Minimum crop size as a fraction of the original size.
        max_aspect: Maximum aspect ratio for crop. Cannot be more than 1/min_crop.
        multiple: The crop size is rounded down to the specified integer multiple.
        distribution: 'flat' or 'exp-log'. How aspect ratios are distributed. If 'flat', then distribution is random uniform
            between (1, max_aspect) and half of time is flipped. If 'exp-log', then distribution is exp of log of uniform
            distribution over (1/max_aspect, max_aspect). 'exp-log' is more biased towards square aspect ratios.

    Returns: 
        Tuple (**X**, **Y**), where

        - **X** - Batch of cropped AFM images of shape ``(batch, x_new, y_new, z)``.
        - **Y** - Batch of cropped reference image descriptors of shape ``(batch, x_new, y_new)``.
    """
    assert 0 < min_crop <= 1.0
    assert max_aspect >= 1.0
    assert 1 / min_crop >= max_aspect

    if distribution == "flat":
        aspect = np.random.uniform(1, max_aspect)
        if np.random.rand() > 0.5:
            aspect = 1 / aspect
    elif distribution == "exp-log":
        aspect = np.exp(np.random.uniform(np.log(1 / max_aspect), np.log(max_aspect)))
    else:
        raise ValueError(f"Unrecognized aspect ratio distribution {distribution}")

    x_size, y_size = X[0].shape[1], X[0].shape[2]
    if aspect > 1.0:
        height = int(np.random.uniform(int(min_crop * y_size), int(y_size / aspect)))
        width = int(height * aspect)
    else:
        width = int(np.random.uniform(int(min_crop * x_size), int(x_size * aspect)))
        height = int(width / aspect)

    width = width - (width % multiple)
    height = height - (height % multiple)

    start_x = int(np.random.uniform(0, x_size - width - 1e-6))
    start_y = int(np.random.uniform(0, y_size - height - 1e-6))

    X = [x[:, start_x : start_x + width, start_y : start_y + height] for x in X]
    Y = [y[:, start_x : start_x + width, start_y : start_y + height] for y in Y]

    return X, Y
