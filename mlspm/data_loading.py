import copy
import io
import os
import random
import re
import time
from itertools import islice
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Tuple

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import get_worker_info

from .graph._data_loading import collate_graph
from .logging import setup_file_logger
from .utils import elements


class ShardList(wds.shardlists.IterableDataset):
    """
    A webdataset shardlist that fills the size of the url list to be divisible by the world size and
    splits the urls by rank. The filling is done by randomly doubling elements in the url list.

    Additionally, can yield random parameters sets for the same shard, using the
    pattern ``*K-{num}*`` in the files names, substituting different numbers for *num*.

    Arguments:
        urls: URLs as a list or brace notation string.
        base_path: The URL paths are relative to this path. Leave empty to use absolute paths.
        world_size: Number of parallel processes over which the URLs are split.
        substitute_param: Split shards into parameter sets and yield random parameter set for each shard.
        log: If not None, path to a file where the yielded shard urls are logged.
    """

    def __init__(
        self,
        urls: list[str] | str,
        base_path: str = "",
        world_size: int = 1,
        rank: int = 0,
        substitute_param: bool = False,
        log: Optional[str] = None,
    ):
        super().__init__()
        self.urls = wds.shardlists.expand_urls(urls)
        if base_path:
            self.urls = [os.path.join(base_path, url) for url in self.urls]
        self.log = log
        self.world_size = world_size
        self.rank = rank
        self._split_param_sets(substitute_param)

    def _get_filled_urls(self):
        l = len(self.urls)
        urls = copy.deepcopy(self.urls)
        orig_urls = urls.copy()
        rem = l % self.world_size
        if rem != 0:
            k = self.world_size - rem
            while k > 0:
                if k >= l:
                    k_ = l
                    urls += orig_urls
                else:
                    k_ = k
                    urls += random.sample(orig_urls, k=k)
                k -= k_
        return urls

    def _split_param_sets(self, sub):
        if not sub:
            self.urls = [[url] for url in self.urls]
        else:
            urls_ = []
            for url in self.urls:
                name = Path(url).name
                for group in urls_:
                    if _name_match(Path(group[0]).name, name):
                        group.append(url)
                        break
                else:
                    urls_.append([url])
            self.urls = urls_

    def _log(self, url):
        setup_file_logger(self.log, "Shards").info(f"{self.rank}, {wi.id if (wi := get_worker_info()) is not None else 0}: {url}")

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        urls = self._get_filled_urls()
        for url in islice(urls, self.rank, None, self.world_size):
            url = random.choice(url)
            if self.log is not None:
                self._log(url)
            yield dict(url=url)


def _name_match(name1, name2):
    m1, m2 = [re.search("K-[0-9]+", name) for name in [name1, name2]]
    for m, name in zip([m1, m2], [name1, name2]):
        if m is None:
            raise ValueError(f"Invalid file name `{name}` does not contain parameter string (K-*).")
    span1, span2 = [range(*m.span()) for m in [m1, m2]]
    name1 = [c for i, c in enumerate(name1) if i not in span1]
    name2 = [c for i, c in enumerate(name2) if i not in span2]
    if name1 != name2:
        return False
    return True


def decode_xyz(key: str, data: Any) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """
    Webdataset pipeline function for decoding xyz files.

    Arguments:
        key: Stream value key. If the key is ``'.xyz'``, then the data is decoded.
        data: Data to decode.

    Returns:
        Tuple (**xyz**, **scan_window**), where

        - **xyz** - Decoded atom coordinates and elements as an array where each row is of the form ``[x, y, z, element]``.
        - **scan_window** - The xyz coordinates of the opposite corners of the scan window in the form
          ``((x_start, y_start, z_start), (x_end, y_end, z_end))``

        If the stream key did not match, the tuple is ``(None, None)`` instead.
    """
    if key == ".xyz":
        data = io.BytesIO(data)
        atom_number = data.readline().decode("utf-8")
        comment = data.readline().decode("utf-8")
        sw = get_scan_window_from_comment(comment)
        xyz = []
        while line := data.readline().decode("utf-8"):
            e, x, y, z = line.strip().split()[:4]
            try:
                e = int(e)
            except ValueError:
                e = elements.index(e) + 1
            xyz.append([np.float32(x), np.float32(y), np.float32(z), e])
        return np.array(xyz).astype(np.float32), sw
    else:
        return None, None


def get_scan_window_from_comment(comment: str) -> np.ndarray:
    """
    Process the comment line in a .xyz file and extract the bounding box of the scan.
    The comment either has the format (QUAM dataset)

        ``Lattice="x0 x1 x2 y0 y1 y2 z0 z1 z2"``

    where the lattice is assumed to be orthogonal and origin at zero, or

        ``Scan window: [[x_start y_start z_start], [x_end y_end z_end]]``

    Arguments:
        comment: Comment to parse.

    Returns:
        The xyz coordinates of the opposite corners of the scan window in the form
            ``((x_start, y_start, z_start), (x_end, y_end, z_end))``
    """
    comment = comment.lower()
    match = re.match('.*lattice="((?:[+-]?(?:[0-9]*\.)?[0-9]+\s?){9})"', comment)
    if match:
        vectors = np.array([float(s) for s in match.group(1).split()])
        vectors = vectors.reshape((3, 3))
        sw = np.zeros((2, 3), dtype=np.float32)
        sw[1] = np.diag(vectors)
    elif match := re.match(
        r".*scan window: [\[(]{2}\s*((?:[+-]?(?:[0-9]*\.)?[0-9]+(?:e[-+]?[0-9]+)?,?\s*){3})[\])],\s*[\[(]\s*((?:[+-]?(?:[0-9]*\.)?[0-9]+(?:e[-+]?[0-9]+)?,?\s*){3})[\])]{2}.*",
        comment,
    ):
        start = np.array([float(s.strip(',')) for s in match.group(1).split()])
        end = np.array([float(s.strip(',')) for s in match.group(2).split()])
        sw = np.stack([start, end], axis=0)
    else:
        raise ValueError(f"Could not parse scan window in comment: `{comment}`")
    return sw


def _rotate_and_stack(src: Iterable[dict], reverse: bool = False) -> Generator[dict, None, None]:
    """
    Take a sample in dict format and update it with fields containing an image stack, xyz coordinates and scan window.
    Rotate the images to be xy-indexing convention and stack them into a single array.

    Likely you don't want to use this function directly, but the wrapper :data:`.rotate_and_stack`.

    Arguments:
        src: Iterable of dicts with the fields:

            - ``'{000..0xx}.{jpg,png}'`` - :class:`PIL.Image.Image` of one slice of the simulation.
            - ``'xyz'`` - Tuple(:class:`np.ndarray`, :class:`np.ndarray`) of the xyz data and the scan window.
            - ``'desc_{0..x}.npy'`` - optional :class:`np.ndarray` of image descriptors.

        reverse: Whether the order of the image stack is reversed.

    Returns:
        Generator that yields sample dicts with the updated fields ``'X'``, ``'Y'``, ``'xyz'``, ``'sw'``.
    """
    for sample in src:
        X, Y, xyz, sw = [], [], None, None
        img_keys = []
        for key in sample.keys():
            if key[-3:] in ["jpg", "png"]:
                num = key[:-4]
                if '.' in num:
                    iz, channel = [int(n) for n in num.split('.')]
                else:
                    iz, channel = int(num), 0
                for _ in range((channel + 1) - len(X)):
                    X.append([])
                X[channel].append((iz, sample[key].rotate(-90)))
                img_keys.append(key)
            elif key[-3:] == "npy":
                match = re.match(r".*desc_(\d+)*.", key)
                i_desc = int(match.group(1))
                Y.append((i_desc, sample[key]))
            elif key == "xyz":
                xyz, sw = sample[key]

        X_ = []
        for x in X:
            # For every channel, stack the z-dimension images into the last dimension
            x = sorted(x, key=(lambda v: v[0]), reverse=reverse)
            x = [v[1] for v in x]
            x = np.stack(x, axis=-1).astype(np.float32)
            X_.append(x)
        X = np.stack(X_, axis=0)

        if len(Y) > 0:
            Y = sorted(Y, key=(lambda v: v[0]), reverse=reverse)
            Y = [v[1] for v in Y]
            Y = np.stack(Y, axis=0)

        sw = np.expand_dims(sw, axis=0)

        for key in img_keys:
            del sample[key]

        sample["X"] = X
        sample["Y"] = Y
        sample["xyz"] = xyz
        sample["sw"] = sw

        yield sample


rotate_and_stack = wds.pipelinefilter(_rotate_and_stack)
"""Webdataset pipeline filter for :func:`_rotate_and_stack`"""


def _collate_batch(batch: Iterable[dict]):
    samples = {}
    for b in batch:
        for key, val in b.items():
            if key in samples:
                samples[key].append(val)
            else:
                samples[key] = [val]

    Xs = samples["X"]
    Ys = samples.get("Y", [])
    sws = samples["sw"]

    # Switch the batch and channel dimension around
    Xs = list(np.stack(Xs, axis=0).transpose(1, 0, 2, 3, 4))
    if len(Ys) > 0 and len(Ys[0]) > 0:
        Ys = list(np.stack(Ys, axis=0).transpose(1, 0, 2, 3))
    sws = list(np.stack(sws, axis=0).transpose(1, 0, 2, 3))

    samples["X"] = Xs
    samples["Y"] = Ys
    samples["sw"] = sws

    return samples


def batched(batch_size: int) -> wds.filters.RestCurried:
    """
    Wrapper for :func:`webdataset.batched` with a suitable collation function.

    The collation function takes collections of sample dictionaries with the following keys and collects them into batched sample
    dictionaries with the same keys:

    - ``'X'`` - AFM images.
    - ``'sw'`` - Scan windows that determine the real-space extent of the AFM image region.
    - ``'Ys'`` - (Optional) Auxiliary image descriptors corresponding to the AFM images.

    Rest of the keys in the dictionary are simply gathered into lists.
    """
    return wds.batched(batch_size, _collate_batch)


def default_collate(batch: Tuple[np.ndarray, ...]) -> Tuple[torch.Tensor, ...]:
    """
    Transfer a batch of Numpy arrays into Pytorch tensors.

    Arguments:
        batch: Should contain at least two arrays (``X``, ``Y``, ...), where ``X`` are AFM images and ``Y`` are image descriptors.

    Returns:
        Tuple (``X``, ``Y``, ...), where the ``X`` and ``Y`` are the AFM images and image descriptors as tensors, and the rest of
        the elements are passed through unchanged.
    """
    X, Y, *rest = batch
    X = [torch.from_numpy(x).unsqueeze(1).float() for x in X]
    Y = [torch.from_numpy(y).float() for y in Y]
    return X, Y, *rest


def worker_init_fn(worker_id: int):
    """
    Initialize each worker with a unique random seed based on it's ID and current time.

    Arguments:
        worker_id: ID of the worker process.
    """
    seed = int((time.time() % 1e5) * 1000) + worker_id
    np.random.seed(seed)
    random.seed(seed)
