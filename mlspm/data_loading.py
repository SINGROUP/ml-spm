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

from .logging import setup_file_logger
from .utils import elements


class ShardList(wds.shardlists.IterableDataset):
    """
    A webdataset shardlist that fills the size of the url list to be divisible by the world size and
    splits the urls by rank. The filling is done by randomly doubling elements in the url list.

    Additionally, can yield random parameters sets for the same shard, using the
    pattern *K-{num}* in the files names, substituting different numbers for num.

    Arguments:
        urls: A list of URLs as a Python list or brace notation string.
        base_path: The URL paths are relative to this path. Leave empty to use absolute paths.
        world_size: Number of parallel processes over which the urls are split.
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
        key: Stream value key. If this is '.xyz', then it is decoded.
        data: Data to decode.

    Returns: tuple with the decoded xyz data and scan window.
    """
    if key == ".xyz":
        data = io.BytesIO(data)
        atom_number = data.readline().decode("utf-8")
        comment = data.readline().decode("utf-8")
        sw = get_scan_window_from_comment(comment)
        xyz = []
        while line := data.readline().decode("utf-8"):
            e, x, y, z, _ = line.strip().split()
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
        Lattice="x0 x1 x2 y0 y1 y2 z0 z1 z2"
    where the lattice is assumed to be orthogonal and origin at zero, or
        Scan window: [[x_start y_start z_start], [x_end y_end z_end]]

    Arguments:
        comment: Comment to parse.

    Returns:
        np.ndarray of shape (2, 3). The xyz coordinates of the opposite corners of the scan window.
    """
    comment = comment.lower()
    match = re.match('.*lattice="((?:[+-]?(?:[0-9]*\.)?[0-9]+\s?){9})"', comment)
    if match:
        vectors = np.array([float(s) for s in match.group(1).split()])
        vectors = vectors.reshape((3, 3))
        sw = np.zeros((2, 3), dtype=np.float32)
        sw[1] = np.diag(vectors)
    elif match := re.match(
        ".*scan window: \[\[\s*((?:[+-]?(?:[0-9]*\.)?[0-9]+(?:e[-+]?[0-9]+)?\s*){3})\],\s*\[\s*((?:[+-]?(?:[0-9]*\.)?[0-9]+(?:e[-+]?[0-9]+)?\s*){3})\]\].*",
        comment,
    ):
        start = np.array([float(s) for s in match.group(1).split()])
        end = np.array([float(s) for s in match.group(2).split()])
        sw = np.stack([start, end], axis=0)
    else:
        raise ValueError(f"Could not parse scan window in comment: `{comment}`")
    return sw


def _rotate_and_stack(src: Iterable[dict], reverse: bool = True) -> Generator[dict, None, None]:
    """
    Take a sample in dict format and update it with fields containing an image stack,
    xyz coordinates and scan window. Rotate the images to be xy-indexing convention and
    stack them into a single array.

    Arguments:
        src: Iterable of dicts with the fields:
            {000..0xx}.jpg: PIL.Image.Image of one slice of the simulation.
            xyz: tuple(np.array, np.array) of the xyz data and the scan window.
        reverse: Whether the order of the image stack is reversed.

    Returns: Generator that yields sample dicts with the updated fields 'X', 'xyz', 'sw'.
    """
    for sample in src:
        X, xyz, sw = [], None, None
        img_keys = []
        for key in sample.keys():
            if key[-3:] in ["jpg", "png"]:
                X.append((int(key[:-4]), sample[key].rotate(-90)))
                img_keys.append(key)
            elif key == "xyz":
                xyz, sw = sample[key]
        X = sorted(X, key=(lambda x: x[0]), reverse=reverse)
        X = [v[1] for v in X]
        X = np.stack(X, axis=-1).astype(np.float32)
        X = np.expand_dims(X, axis=0)
        sw = np.expand_dims(sw, axis=0)
        for key in img_keys:
            del sample[key]
        sample["X"] = X
        sample["xyz"] = xyz
        sample["sw"] = sw
        yield sample


rotate_and_stack = wds.pipelinefilter(_rotate_and_stack)


def _collate_batch(batch: Iterable[dict]):
    Xs, xyzs, sws, keys, urls = [[b[k] for b in batch] for k in ["X", "xyz", "sw", "__key__", "__url__"]]
    Ys = [b.get("Y", []) for b in batch]

    Xs = list(np.stack(Xs, axis=0).transpose(1, 0, 2, 3, 4))
    if len(Ys[0]) > 0:
        Ys = list(np.stack(Ys, axis=0).transpose(1, 0, 2, 3))
    sws = list(np.stack(sws, axis=0).transpose(1, 0, 2, 3))

    sample = {"X": Xs, "Y": Ys, "xyz": xyzs, "sw": sws, "__key__": keys, "__url__": urls}

    return sample


def batched(batch_size: int):
    """Wrapper for wds.batched with a suitable collation function."""
    return wds.batched(batch_size, _collate_batch)


def default_collate(batch):
    X, Y, *rest = batch
    X = [torch.from_numpy(x).unsqueeze(1).float() for x in X]
    Y = [torch.from_numpy(y).float() for y in Y]
    return X, Y, *rest


def worker_init_fn(worker_id):
    seed = int((time.time() % 1e5) * 1000) + worker_id
    np.random.seed(seed)
    random.seed(seed)
