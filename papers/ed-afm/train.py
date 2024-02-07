import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import webdataset as wds
import yaml
from torch import optim
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel

import mlspm.data_loading as dl
import mlspm.preprocessing as pp
import mlspm.visualization as vis
from mlspm import utils
from mlspm.cli import parse_args
from mlspm.image.models import EDAFMNet
from mlspm.logging import LossLogPlot, SyncedLoss
from mlspm.losses import WeightedMSELoss


def make_model(device, cfg):
    model = EDAFMNet(device=device)
    criterion = WeightedMSELoss(cfg["loss_weights"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    lr_decay_rate = 1e-5
    lr_decay = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0 / (1.0 + lr_decay_rate * b))
    return model, criterion, optimizer, lr_decay


def apply_preprocessing(batch):

    X = batch["X"]
    Y = batch["Y"]
    xyz = batch["xyz"]

    X = [X[0], X[1]]  # Pick CO and Xe
    X = [x[:, :, :, 2:8] for x in X]
    pp.rand_shift_xy_trend(X, max_layer_shift=0.02, max_total_shift=0.04)

    X, Y = pp.add_rotation_reflection(X, Y, reflections=True, multiple=3, crop=(128, 128))
    X, Y = pp.random_crop(X, Y, min_crop=0.75, max_aspect=1.25)

    pp.add_norm(X, per_layer=True)
    pp.add_gradient(X, c=0.3)
    pp.add_noise(X, c=0.1, randomize_amplitude=True, normal_amplitude=True)
    pp.add_cutout(X, n_holes=5)

    return X, Y, xyz


def make_webDataloader(cfg, mode="train"):
    assert mode in ["train", "val", "test"], mode

    shard_list = dl.ShardList(
        cfg[f"urls_{mode}"],
        base_path=cfg["data_dir"],
        world_size=cfg["world_size"],
        rank=cfg["global_rank"],
        substitute_param=(mode == "train"),
        log=Path(cfg["run_dir"]) / "shards.log",
    )

    dataset = wds.WebDataset(shard_list)
    dataset.pipeline.pop()
    if mode == "train":
        dataset.append(wds.shuffle(10))  # Shuffle order of shards
    dataset.append(wds.tariterators.tarfile_to_samples())  # Gather files inside tar files into samples
    dataset.append(wds.split_by_worker)  # Use a different subset of samples in shards in different workers
    if mode == "train":
        dataset.append(wds.shuffle(100))  # Shuffle samples within a worker process
    dataset.append(wds.decode("pill", wds.autodecode.basichandlers, dl.decode_xyz))  # Decode image and xyz files
    dataset.append(dl.rotate_and_stack())  # Combine separate images into a stack
    dataset.append(dl.batched(cfg["batch_size"]))  # Gather samples into batches
    dataset = dataset.map(apply_preprocessing)  # Preprocess batch

    dataloader = wds.WebLoader(
        dataset,
        num_workers=cfg["num_workers"],
        batch_size=None,  # Batching is done in the WebDataset
        pin_memory=True,
        collate_fn=dl.default_collate,
        persistent_workers=False,
    )

    return dataset, dataloader


def batch_to_device(batch, device):
    X, ref, xyz = batch
    X = [x.to(device) for x in X]
    ref = [y.to(device) for y in ref]
    return X, ref, xyz


def run(cfg):
    print(f'Starting on global rank {cfg["global_rank"]}, local rank {cfg["local_rank"]}\n', flush=True)

    # Initialize the distributed environment.
    dist.init_process_group(cfg["comm_backend"])

    start_time = time.perf_counter()

    if cfg["global_rank"] == 0:
        # Create run directory
        if not os.path.exists(cfg["run_dir"]):
            os.makedirs(cfg["run_dir"])
    dist.barrier()

    # Define model, optimizer, and loss
    model, criterion, optimizer, lr_decay = make_model(cfg["local_rank"], cfg)

    print(f'World size = {cfg["world_size"]}')
    print(f"Trainable parameters: {utils.count_parameters(model)}")

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

    # Wrap model in DistributedDataParallel.
    model = DistributedDataParallel(model, device_ids=[cfg["local_rank"]], find_unused_parameters=False)

    if cfg["train"]:
        # Create datasets and dataloaders
        _, train_loader = make_webDataloader(cfg, "train")
        _, val_loader = make_webDataloader(cfg, "val")

        if cfg["global_rank"] == 0:
            if init_epoch <= cfg["epochs"]:
                print(f"\n ========= Starting training from epoch {init_epoch}")
            else:
                print("Model already trained")

        for epoch in range(init_epoch, cfg["epochs"] + 1):

            if cfg["global_rank"] == 0:
                print(f"\n === Epoch {epoch}")

            # Train
            if cfg["timings"] and cfg["global_rank"] == 0:
                t0 = time.perf_counter()

            model.train()
            with Join([model, loss_logger.get_joinable("train")]):
                for ib, batch in enumerate(train_loader):

                    # Transfer batch to device
                    X, ref, _ = batch_to_device(batch, cfg["local_rank"])

                    if cfg["timings"] and cfg["global_rank"] == 0:
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()

                    # Forward
                    pred, _ = model(X)
                    losses = criterion(pred, ref)
                    loss = losses[0]

                    if cfg["timings"] and cfg["global_rank"] == 0:
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_decay.step()

                    # Log losses
                    loss_logger.add_train_loss(loss)

                    if cfg["timings"] and cfg["global_rank"] == 0:
                        torch.cuda.synchronize()
                        t3 = time.perf_counter()
                        print(f"(Train) Load Batch/Forward/Backward: {t1-t0:6f}/{t2-t1:6f}/{t3-t2:6f}")
                        t0 = t3

            # Validate
            if cfg["global_rank"] == 0:
                val_start = time.perf_counter()
                if cfg["timings"]:
                    t0 = val_start

            model.eval()
            with Join([loss_logger.get_joinable("val")]):
                with torch.no_grad():
                    for ib, batch in enumerate(val_loader):

                        # Transfer batch to device
                        X, ref, _ = batch_to_device(batch, cfg["local_rank"])

                        if cfg["timings"] and cfg["global_rank"] == 0:
                            torch.cuda.synchronize()
                            t1 = time.perf_counter()

                        # Forward
                        pred, _ = model(X)
                        losses = criterion(pred, ref)

                        loss_logger.add_val_loss(losses[0])

                        if cfg["timings"] and cfg["global_rank"] == 0:
                            torch.cuda.synchronize()
                            t2 = time.perf_counter()
                            print(f"(Val) Load Batch/Forward: {t1-t0:6f}/{t2-t1:6f}")
                            t0 = t2

            # Write average losses to log and report to terminal
            loss_logger.next_epoch()

            # Save checkpoint
            checkpointer.next_epoch(loss_logger.val_losses[-1][0])

    # Return to best epoch
    checkpointer.revert_to_best_epoch()

    # Return to best epoch, and save model weights
    dist.barrier()
    checkpointer.revert_to_best_epoch()
    if cfg["global_rank"] == 0:
        torch.save(model.module.state_dict(), save_path := os.path.join(cfg["run_dir"], "best_model.pth"))
        print(f"\nModel saved to {save_path}")
        print(f"Best validation loss on epoch {checkpointer.best_epoch}: {checkpointer.best_loss}")
        print(
            f'Average of best {cfg["avg_best_epochs"]} validation losses: '
            f'{np.sort(loss_logger.val_losses[:, 0])[:cfg["avg_best_epochs"]].mean()}'
        )

    if cfg["test"] or cfg["predict"]:
        _, test_loader = make_webDataloader(cfg, "test")

    if cfg["test"]:

        if cfg["global_rank"] == 0:
            print(f"\n ========= Testing with model from epoch {checkpointer.best_epoch}")

        eval_losses = SyncedLoss(num_losses=len(loss_logger.loss_labels))
        eval_start = time.perf_counter()
        if cfg["timings"] and cfg["global_rank"] == 0:
            t0 = eval_start

        model.eval()
        with Join([eval_losses]):
            with torch.no_grad():
                for ib, batch in enumerate(test_loader):

                    # Transfer batch to device
                    X, ref, xyz = batch_to_device(batch, cfg["local_rank"])

                    if cfg["timings"] and cfg["global_rank"] == 0:
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()

                    # Forward
                    pred, _ = model(X)
                    losses = criterion(pred, ref)
                    eval_losses.append(losses[0])

                    if (ib + 1) % cfg["print_interval"] == 0:
                        print(f"Test Batch {ib+1}")

                    if cfg["timings"] and cfg["global_rank"] == 0:
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        print(f"(Test) t0/Load Batch/Forward: {t1-t0:6f}/{t2-t1:6f}/")
                        t0 = t2

        # Average losses and print
        eval_loss = eval_losses.mean()
        print(f"Test set loss: {loss_logger.loss_str(eval_loss)}")

        # Save test set loss to file
        with open(os.path.join(cfg["run_dir"], "test_loss.txt"), "w") as f:
            f.write(";".join([str(l) for l in eval_loss]))

    if cfg["predict"] and cfg["global_rank"] == 0:

        # Make predictions
        print(f'\n ========= Predict on {cfg["pred_batches"]} batches from the test set')
        counter = 0
        pred_dir = os.path.join(cfg["run_dir"], "predictions/")

        with torch.no_grad():
            for ib, batch in enumerate(test_loader):

                if ib >= cfg["pred_batches"]:
                    break

                # Transfer batch to device
                X, ref, xyz = batch_to_device(batch, cfg["local_rank"])

                # Forward
                pred, _ = model(X)

                # Data back to host
                X = [x.squeeze(1).cpu().numpy() for x in X]
                pred = [p.cpu().numpy() for p in pred]
                ref = [r.cpu().numpy() for r in ref]

                # Save xyzs
                utils.batch_write_xyzs(xyz, outdir=pred_dir, start_ind=counter)

                # Visualize input AFM images and predictions
                vis.make_input_plots(X, outdir=pred_dir, start_ind=counter)
                vis.make_prediction_plots(pred, ref, descriptors=cfg["loss_labels"], outdir=pred_dir, start_ind=counter)

                counter += len(X[0])

    print(f'Done at rank {cfg["global_rank"]}. Total time: {time.perf_counter() - start_time:.0f}s')

    log_file.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    # Get config
    cfg = parse_args()
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(exist_ok=True, parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        # Remember settings
        yaml.safe_dump(cfg, f)

    # Set random seeds
    torch.manual_seed(cfg["random_seed"])
    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])

    # Start run
    cfg["world_size"] = int(os.environ["WORLD_SIZE"])
    cfg["global_rank"] = int(os.environ["RANK"])
    cfg["local_rank"] = int(os.environ["LOCAL_RANK"])
    run(cfg)
