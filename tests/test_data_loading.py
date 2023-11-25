import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def test_shardlist():
    from mlspm.data_loading import ShardList

    urls = "test_K-{0..1}_{0..1}"
    base_path = "./base/"
    log_path = Path("shards.log")

    # Test with subtitution
    shard_list = ShardList(urls=urls, base_path=base_path, world_size=1, rank=0, substitute_param=True, log=log_path)

    assert len(list(shard_list)) == 2

    # Test without subtitution
    shard_list = ShardList(urls=urls, base_path=base_path, world_size=1, rank=0, substitute_param=False, log=log_path)

    shards = list(shard_list)
    shards_expected = [
        {"url": "./base/test_K-0_0"},
        {"url": "./base/test_K-0_1"},
        {"url": "./base/test_K-1_0"},
        {"url": "./base/test_K-1_1"},
    ]
    assert shards == shards_expected

    # Test splitting over ranks
    shard_list_rank0 = ShardList(urls=urls, base_path=base_path, world_size=2, rank=0, substitute_param=False, log=log_path)
    shard_list_rank1 = ShardList(urls=urls, base_path=base_path, world_size=2, rank=1, substitute_param=False, log=log_path)

    shards = list(shard_list_rank0)
    shards_expected = [
        {"url": "./base/test_K-0_0"},
        {"url": "./base/test_K-1_0"},
    ]
    assert shards == shards_expected

    shards = list(shard_list_rank1)
    shards_expected = [
        {"url": "./base/test_K-0_1"},
        {"url": "./base/test_K-1_1"},
    ]
    assert shards == shards_expected

    # Test url list filling (list size not divisible by world_size)
    shard_list_rank0 = ShardList(urls=urls, base_path=base_path, world_size=3, rank=0, substitute_param=False, log=log_path)
    shard_list_rank1 = ShardList(urls=urls, base_path=base_path, world_size=3, rank=1, substitute_param=False, log=log_path)
    shard_list_rank2 = ShardList(urls=urls, base_path=base_path, world_size=3, rank=2, substitute_param=False, log=log_path)

    assert len(list(shard_list_rank0)) == 2
    assert len(list(shard_list_rank1)) == 2
    assert len(list(shard_list_rank2)) == 2

    # Test url list filling (world size at least twice as big as the list size)
    shard_list = ShardList(urls=urls, base_path=base_path, world_size=9, rank=8, substitute_param=False, log=log_path)
    assert len(list(shard_list)) == 1

    with pytest.raises(ValueError):
        # With substitute_param=True, requires "K_{num}" in urls
        shard_list = ShardList(urls="test_{0..1}", substitute_param=True, log=log_path)
        list(shard_list)

    if log_path.exists():
        os.remove(log_path)


def test_decode_xyz():
    from mlspm.data_loading import decode_xyz
    from mlspm.utils import read_xyzs, write_to_xyz

    xyz_path = "./test_decode.xyx"
    # fmt: off
    xyz = np.array([
        [ 0.0,  0.1,  0.2,  1],  # (x, y, z, element)
        [ 0.0, -0.1, -0.2,  6],
        [-0.2,  1.5, -2.5, 14]]
    )
    # fmt: on

    write_to_xyz(xyz, xyz_path, comment_str="Scan window: [[-1.0 1.0 .2], [10.0 20 15.0]]`")
    with open(xyz_path, "rb") as f:
        xyz_bytes = f.read()

    os.remove(xyz_path)

    xyz_decoded, scan_window = decode_xyz(".xyz", data=xyz_bytes)

    assert np.allclose(xyz, xyz_decoded)
    assert np.allclose(scan_window, np.array([[-1.0, 1.0, 0.2], [10.0, 20.0, 15.0]]))

    a, b = decode_xyz(".png", data=xyz_bytes)

    assert a is None
    assert b is None


def test_rotate_and_stack():
    from mlspm.data_loading import _rotate_and_stack

    img0 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    img1 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    img0_pil = Image.fromarray(img0)
    img1_pil = Image.fromarray(img1)

    # fmt: off
    xyz = np.array([
        [ 0.0,  0.1,  0.2,  1],  # (x, y, z, element)
        [ 0.0, -0.1, -0.2,  6],
        [-0.2,  1.5, -2.5, 14]]
    )
    # fmt: on
    sw = np.array([[-1.0, 1.0, 0.2], [10.0, 20.0, 15.0]])

    src = [{"000.png": img0_pil, "001.png": img1_pil, "xyz": (xyz, sw)}]

    samples = list(_rotate_and_stack(src, reverse=False))

    sample = samples[0]
    X = sample["X"]
    xyz_out = sample["xyz"]
    sw_out = sample["sw"]

    assert X.shape == (1, 10, 10, 2)
    assert sw_out.shape == (1, 2, 3)
    assert xyz_out.shape == (3, 4)

    assert np.allclose(X[0, :, ::-1, 0].T, img0)
    assert np.allclose(X[0, :, ::-1, 1].T, img1)
    assert np.allclose(xyz, xyz_out)
    assert np.allclose(sw, sw_out)

def test_collate_batch():

    from mlspm.data_loading import _collate_batch

    batch = [
        {
            'X': np.random.randint(0, 255, (1, 10, 10, 5), dtype=np.uint8),
            'Y': np.random.randint(0, 255, (1, 10, 10), dtype=np.uint8),
            'xyz': np.random.rand(3, 4),
            'sw': np.random.rand(1, 2, 3)
        }
        for _ in range(3)
    ]

    batch = _collate_batch(batch)

    X = batch['X']
    Y = batch['Y']
    xyz = batch['xyz']
    sw = batch['sw']

    assert len(X) == len(Y) == len(sw) == 1
    assert X[0].shape == (3, 10, 10, 5)
    assert Y[0].shape == (3, 10, 10)
    assert sw[0].shape == (3, 2, 3)
    assert len(xyz) == 3
    for a in xyz:
        assert a.shape == (3, 4)
