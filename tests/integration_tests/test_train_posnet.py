#!/usr/bin/env python3

import os
import random
import shutil
import tarfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch import nn, optim

import mlspm.data_loading as dl
import mlspm.preprocessing as pp
from mlspm import graph, utils
from mlspm.cli import parse_args
from mlspm.logging import LossLogPlot
from mlspm.models import PosNet

from PIL import Image


def make_model(device, cfg):
    outsize = round((cfg["z_lims"][1] - cfg["z_lims"][0]) / cfg["box_res"][2]) + 1
    model = PosNet(
        encode_block_channels=[2, 4, 8, 16],
        encode_block_depth=2,
        decode_block_channels=[16, 8, 4],
        decode_block_depth=1,
        decode_block_channels2=[16, 8, 4],
        decode_block_depth2=1,
        attention_channels=[16, 16, 16],
        res_connections=True,
        activation="relu",
        padding_mode="zeros",
        pool_type="avg",
        decoder_z_sizes=[5, 10, outsize],
        z_outs=[3, 3, 5, 8],
        peak_std=cfg["peak_std"],
        device=device
    )
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    lr_decay_rate = 1e-5
    lr_decay = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0 / (1.0 + lr_decay_rate * b))
    return model, criterion, optimizer, lr_decay


def make_test_data(cfg):
    out_dir = Path(cfg["data_dir"])
    out_dir.mkdir(exist_ok=True)
    urls = wds.shardlists.expand_urls(cfg["urls_train"])
    i_sample = 0
    for url in urls:
        temp_dir = Path(f"temp_{url}")
        temp_dir.mkdir(exist_ok=True)
        os.chdir(temp_dir)
        with tarfile.open(url, "w") as f:
            for _ in range(10):
                afm = np.random.randint(0, 255, (64, 64, 8), dtype=np.uint8)
                for i in range(afm.shape[-1]):
                    img_path = f"{i_sample}.{i}.png"
                    Image.fromarray(afm[:, ::-1, i].T).save(img_path)
                    f.add(img_path)
                xyz = np.random.rand(8, 3)
                xyz[:, :2] *= 8
                atoms = np.concatenate([xyz, np.random.randint(1, 10, (8, 1))], axis=1)
                xyz_path = f"{i_sample}.xyz"
                utils.write_to_xyz(atoms, outfile=xyz_path, comment_str="Scan window: [[0.0 0.0 0.0], [8.0 8.0 1.0]]", verbose=0)
                f.add(xyz_path)
                i_sample += 1
        os.chdir("..")
        (temp_dir / url).rename(out_dir / url)
        shutil.rmtree(temp_dir)


def apply_preprocessing(batch, cfg):
    box_res = cfg["box_res"]
    z_lims = cfg["z_lims"]
    zmin = cfg["zmin"]
    peak_std = cfg["peak_std"]

    X, atoms, scan_windows = [batch[k] for k in ["X", "xyz", "sw"]]

    nz_max = X[0].shape[-1]
    nz = random.choice(range(1, nz_max + 1))
    z0 = random.choice(range(0, min(5, nz_max + 1 - nz)))
    X = [x[:, :, :, -nz:] for x in X] if z0 == 0 else [x[:, :, :, -(nz + z0) : -z0] for x in X]

    atoms = [a[a[:, -1] != 29] for a in atoms]
    pp.top_atom_to_zero(atoms)
    xyz = atoms.copy()
    mols = [graph.MoleculeGraph(a, []) for a in atoms]
    mols, sw = graph.shift_mols_window(mols, scan_windows[0])

    pp.rand_shift_xy_trend(X, max_layer_shift=0.02, max_total_shift=0.04)
    box_borders = graph.make_box_borders(X[0].shape[1:3], res=box_res[:2], z_range=z_lims)
    X, mols, box_borders = graph.add_rotation_reflection_graph(
        X, mols, box_borders, num_rotations=1, reflections=True, crop=(32, 32), per_batch_item=True
    )
    pp.add_norm(X)
    pp.add_gradient(X, c=0.3)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.add_cutout(X, n_holes=5)

    mols = graph.threshold_atoms_bonds(mols, zmin)
    ref = graph.make_position_distribution(mols, box_borders, box_res=box_res, std=peak_std)

    return X, [ref], xyz, box_borders


def make_webDataloader(cfg):
    shard_list = dl.ShardList(
        cfg[f"urls_train"],
        base_path=cfg["data_dir"],
        substitute_param=True,
        log=Path(cfg["run_dir"]) / "shards.log",
    )

    dataset = wds.WebDataset(shard_list)
    dataset.pipeline.pop()
    dataset.append(wds.tariterators.tarfile_to_samples())
    dataset.append(wds.split_by_worker)
    dataset.append(wds.decode("pill", dl.decode_xyz))
    dataset.append(dl.rotate_and_stack())
    dataset.append(dl.batched(cfg["batch_size"]))
    dataset = dataset.map(partial(apply_preprocessing, cfg=cfg))

    dataloader = wds.WebLoader(
        dataset,
        num_workers=cfg["num_workers"],
        batch_size=None,
        pin_memory=True,
        collate_fn=dl.default_collate,
        persistent_workers=False,
    )

    return dataset, dataloader


def batch_to_device(batch, device):
    X, ref, *rest = batch
    X = X[0].to(device)
    ref = ref[0].to(device)
    return X, ref, *rest


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create run directory
    if not os.path.exists(cfg["run_dir"]):
        os.makedirs(cfg["run_dir"])

    # Define model, optimizer, and loss
    model, criterion, optimizer, lr_decay = make_model(device, cfg)

    # Setup checkpointing and load a checkpoint if available
    checkpointer = utils.Checkpointer(
        model,
        optimizer,
        additional_data={"lr_params": lr_decay},
        checkpoint_dir=os.path.join(cfg["run_dir"], "Checkpoints/"),
        keep_last_epoch=True,
    )
    init_epoch = checkpointer.epoch

    # Setup logging
    log_file = open(os.path.join(cfg["run_dir"], "batches.log"), "a")
    loss_logger = LossLogPlot(
        log_path=os.path.join(cfg["run_dir"], "loss_log.csv"),
        plot_path=os.path.join(cfg["run_dir"], "loss_history.png"),
        loss_labels=cfg["loss_labels"],
        loss_weights=cfg["loss_weights"],
        print_interval=cfg["print_interval"],
        init_epoch=init_epoch,
        stream=log_file,
    )

    for epoch in range(cfg["epochs"]):
        # Create datasets and dataloaders
        _, train_loader = make_webDataloader(cfg)
        val_loader = train_loader

        print(f"\n === Epoch {epoch}")

        model.train()
        for ib, batch in enumerate(train_loader):
            # Transfer batch to device
            X, ref, _, _ = batch_to_device(batch, device)

            # Forward
            pred, _ = model(X)
            loss = criterion(pred, ref)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_decay.step()

            # Log losses
            loss_logger.add_train_loss(loss)

            print(f"Train batch {ib}")

        # Validate

        model.eval()
        with torch.no_grad():
            for ib, batch in enumerate(val_loader):
                # Transfer batch to device
                X, ref, _, _ = batch_to_device(batch, device)

                # Forward
                pred, _ = model(X)
                loss = criterion(pred, ref)

                loss_logger.add_val_loss(loss)

                print(f"Val batch {ib}")

        # Write average losses to log and report to terminal
        loss_logger.next_epoch()

        # Save checkpoint
        checkpointer.next_epoch(loss_logger.val_losses[-1][0])

    # Return to best epoch, and save model weights
    checkpointer.revert_to_best_epoch()
    print(f"Best validation loss on epoch {checkpointer.best_epoch}: {checkpointer.best_loss}")

    log_file.close()
    shutil.rmtree(cfg["run_dir"])
    shutil.rmtree(cfg["data_dir"])


def test_train_posnet():
    # fmt:off
    cfg = parse_args(
        [
            "--run_dir", "test_train",
            "--epochs", "2",
            "--batch_size", "4",
            "--z_lims", "-1.0", "0.5",
            "--zmin", "-1.0",
            "--data_dir", "./test_data",
            "--urls_train", "train-K-{0..1}_{0..1}.tar",
            "--box_res", "0.125", "0.125", "0.100",
            "--peak_std", "0.20",
            "--lr", "1e-4"
        ]
    )
    # fmt:on

    make_test_data(cfg)
    run(cfg)


if __name__ == "__main__":
    test_train_posnet()
