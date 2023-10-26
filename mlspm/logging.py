import datetime
import logging
import os
import sys
import time
from typing import Optional, TextIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook

from .utils import _calc_plot_dim, _get_distributed


def setup_file_logger(save_path: str, logger_name: str, first_line: str = ""):
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    f_handler = logging.FileHandler(save_path)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.info(first_line)
    return logger


class SyncedLoss(Joinable):
    """Gather loss values to a list that is averaged over parallel ranks.

    Arguments:
        num_losses: Number of different loss values.
    """

    def __init__(self, num_losses: int):
        super().__init__()
        self.num_losses = num_losses
        self.world_size, self.local_rank, self.global_rank, self.group = _get_distributed()
        self.reset()

    def reset(self):
        """Empty list of losses"""
        self.losses = []
        self.n_batches = 0

    def __len__(self):
        return len(self.losses)

    def __getitem__(self, index: int):
        return self.losses[index]

    def mean(self) -> np.ndarray:
        """Get average loss over batches."""
        return np.mean(self.losses, axis=0)

    @property
    def join_process_group(self):
        return self.group

    @property
    def join_device(self):
        return self.local_rank

    def join_hook(self, **kwargs):
        return SyncedLossJoinHook(self)

    def _sync_losses(self, losses, shadow=False):
        assert self.world_size > 1, self.world_size

        if not shadow:
            assert len(losses) == self.num_losses, losses

            # We haven't joined yet
            Join.notify_join_context(self)

            # Count non-joined ranks
            world_size_eff = torch.ones(1, device=self.local_rank)
            dist.all_reduce(world_size_eff, op=dist.ReduceOp.SUM)

            # Sum losses over non-joined ranks
            losses = torch.tensor(losses, device=self.local_rank)
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)

        else:  # We joined already, so shadow the reduce operations
            # Don't count towards non-joined ranks
            world_size_eff = torch.zeros(1, device=self.local_rank)
            dist.all_reduce(world_size_eff, op=dist.ReduceOp.SUM)

            # Also don't count towards sum of losses
            losses = torch.zeros(self.num_losses, device=self.local_rank)
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)

        # Add averaged losses to list
        losses /= world_size_eff
        losses = list(losses.cpu().numpy())
        self.losses.append(losses)

        return losses

    def append(self, losses: torch.Tensor | np.ndarray | float | list[float]):
        """
        Append a new batch of loss values.

        Arguments:
            losses: Loss values. Length should match self.num_losses.
        """
        if not isinstance(losses, list):
            losses = [losses]
        losses_ = []
        for loss in losses:
            if isinstance(loss, torch.Tensor):
                if loss.size() == ():
                    losses_.append(loss.item())
                else:
                    losses_ += list(loss.cpu().detach().numpy())
            elif isinstance(loss, np.ndarray):
                if loss.size == 1:
                    losses_.append(loss.item())
                else:
                    losses_ += list(loss)
            elif isinstance(loss, (int, float)):
                losses_.append(loss)
            else:
                raise ValueError(f"Loss has unsupported type `{type(loss)}`")
        losses = losses_
        if np.isnan(losses).any():
            raise ValueError(
                f"Found a nan in losses ({losses}) at rank {self.global_rank} after {len(self.losses)} batches. "
                f"Some of the previous losses were: {self.losses[-5:]}"
            )
        if self.world_size > 1:
            losses = self._sync_losses(losses)
        else:
            self.losses.append(losses)
        self.n_batches += 1
        return losses


class SyncedLossJoinHook(JoinHook):
    """Hook for when the number of batches does not match between processes."""

    def __init__(self, synced_loss):
        self.synced_loss = synced_loss

    def main_hook(self):
        self.synced_loss._sync_losses([], shadow=True)

    def post_hook(self, is_last_joiner):
        pass


class LossLogPlot:
    """
    Log and plot model training loss history. Add epoch losses with add_losses and plot with plot_history.
    Works with distributed training.

    Arguments:
        log_path: str. Path where loss log is saved.
        plot_path: str. Path where plot of loss history is saved.
        loss_labels: list of str. Labels for different loss components. If length > 1,
            an additional component 'Total' is prepended to the list.
        loss_weights: Weights for different loss components when there is more than one.
        print_interval: int. Loss values are printed every print_interval batches.
        init_epoch: int or None. Initial epoch. If not None and existing log has more epochs, discard them.
        stream: file object. Stream where log is printed to.
    """

    def __init__(
        self,
        log_path: str,
        plot_path: str,
        loss_labels: list[str],
        loss_weights: Optional[list[float]] = None,
        print_interval: int = 10,
        init_epoch: Optional[int] = None,
        stream: TextIO = sys.stdout,
    ):
        self.log_path = log_path
        self.plot_path = plot_path
        self.print_interval = print_interval
        self.stream = stream

        if len(loss_labels) > 1:
            loss_labels = ["Total"] + loss_labels
        self.loss_labels = loss_labels

        if loss_weights is None or len(loss_weights) == 0:
            loss_weights = [""] * len(self.loss_labels)
        else:
            if len(loss_labels) == 1:
                assert len(loss_weights) == 1
            else:
                assert len(loss_weights) == (len(loss_labels) - 1)
                loss_weights = [""] + loss_weights
        self.loss_weights = loss_weights

        self.train_losses = np.empty((0, len(loss_labels)))
        self.val_losses = np.empty((0, len(loss_labels)))
        self.world_size, self.local_rank, self.global_rank, _ = _get_distributed()
        self.epoch = 1
        self._synced_losses = {"train": SyncedLoss(len(self.loss_labels)), "val": SyncedLoss(len(self.loss_labels))}
        self._init_log(init_epoch)

    def __del__(self):
        if self.stream is not sys.stdout:
            self.stream.close()

    def _init_log(self, init_epoch):
        log_exists = os.path.isfile(self.log_path)
        if self.world_size > 1:
            dist.barrier()
        if not log_exists:
            if self.global_rank > 0:
                return
            self._write_log()
            print(f"Created log at {self.log_path}", file=self.stream, flush=True)
        else:
            with open(self.log_path, "r") as f:
                header = f.readline().rstrip("\r\n").split(";")
                hl = (len(header) - 1) // 2
                if len(self.loss_labels) != hl:
                    raise ValueError(
                        f"The length of the given list of loss names and the length of the header of the existing log at {self.log_path} do not match."
                    )
                for line in f:
                    if init_epoch is not None and self.epoch >= init_epoch:
                        break
                    line = line.rstrip("\n").split(";")
                    if len(line) < 3:
                        continue
                    self.train_losses = np.append(self.train_losses, [[float(s) for s in line[1 : hl + 1]]], axis=0)
                    self.val_losses = np.append(self.val_losses, [[float(s) for s in line[hl + 1 :]]], axis=0)
                    self.epoch += 1
            if self.global_rank == 0:
                if init_epoch is not None:
                    self._write_log()  # Make sure there are no additional rows in the log
                print(f"Using existing log at {self.log_path}", file=self.stream, flush=True)

    def _write_log(self):
        with open(self.log_path, "w") as f:
            f.write("epoch")
            for i, label in enumerate(self.loss_labels):
                label = f";train_{label}"
                if self.loss_weights[i]:
                    label += f" (x {self.loss_weights[i]})"
                f.write(label)
            for i, label in enumerate(self.loss_labels):
                label = f";val_{label}"
                if self.loss_weights[i]:
                    label += f" (x {self.loss_weights[i]})"
                f.write(label)
            f.write("\n")
            for epoch, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
                f.write(str(epoch + 1))
                for l in train_loss:
                    f.write(f";{l}")
                for l in val_loss:
                    f.write(f";{l}")
                f.write("\n")

    def _add_loss(self, losses, mode="train"):
        synced_loss = self._synced_losses[mode]
        losses = synced_loss.append(losses)
        if len(losses) != len(self.loss_labels):
            raise ValueError(f"Length of losses ({len(losses)}) does not match with number of " f"loss labels ({len(self.loss_labels)}).")
        if self.global_rank == 0 and len(synced_loss) % self.print_interval == 0:
            self._print_losses(mode)

    def _print_losses(self, mode="train"):
        if self.global_rank > 0:
            return
        synced_loss = self._synced_losses[mode]
        losses = np.mean(synced_loss[-self.print_interval :], axis=0)
        print(f"Epoch {self.epoch}, {mode} batch {len(synced_loss)} - Loss: " + self.loss_str(losses), file=self.stream, flush=True)

    def loss_str(self, losses: list[float] | np.ndarray | torch.Tensor):
        """Get pretty string for loss values."""
        if len(losses) != len(self.loss_labels):
            raise ValueError(f"Length of losses ({len(losses)}) does not match with number of " f"loss labels ({len(self.loss_labels)}).")
        if len(self.loss_labels) == 1:
            msg = f"{self.loss_labels[0]}: {losses[0]:.6f}"
        else:
            msg = f"{losses[0]:.6f}"
            msg_loss = [f"{label}: {loss:.6f}" for label, loss in zip(self.loss_labels[1:], losses[1:])]
            msg += " (" + ", ".join(msg_loss) + ")"
        return msg

    def add_train_loss(self, losses: torch.Tensor | np.ndarray | float | list[float]):
        """Add losses for one training batch. Averaged over parallel processes.

        Arguments:
            losses: Losses to append to the list.
        """
        if len(self._synced_losses["train"]) == 0:
            self.epoch_start = time.perf_counter()
        self._add_loss(losses, mode="train")

    def add_val_loss(self, losses: torch.Tensor | np.ndarray | float | list[float]):
        """Add losses for one validation batch. Averaged over parallel processes.

        Arguments:
            losses: Losses to append to the list.
        """
        if len(self._synced_losses["val"]) == 0:
            self.val_start = time.perf_counter()
        self._add_loss(losses, mode="val")

    def next_epoch(self):
        """
        Increment epoch by one, write current average batch losses to log, empty batch losses,
        report epoch time to terminal, and update loss history plot.
        """

        train_loss = self._synced_losses["train"].mean()
        val_loss = self._synced_losses["val"].mean()
        self.train_losses = np.append(self.train_losses, train_loss[None], axis=0)
        self.val_losses = np.append(self.val_losses, val_loss[None], axis=0)

        n_train = self._synced_losses["train"].n_batches
        n_val = self._synced_losses["val"].n_batches
        print(
            f"Epoch {self.epoch} at rank {self.global_rank} contained {n_train} training batches " f"and {n_val} validation batches",
            file=self.stream,
            flush=True,
        )

        if self.global_rank == 0:
            epoch_end = time.perf_counter()
            train_step = (self.val_start - self.epoch_start) / n_train
            val_step = (epoch_end - self.val_start) / n_val
            print(f"Completed epoch {self.epoch} at {datetime.datetime.now()}", file=self.stream, flush=True)
            print(f"Train loss: {self.loss_str(train_loss)}", file=self.stream, flush=True)
            print(f"Val loss: {self.loss_str(val_loss)}", file=self.stream, flush=True)
            print(
                f"Epoch time: {epoch_end - self.epoch_start:.2f}s - Train step: {train_step:.5f}s " f"- Val step: {val_step:.5f}s",
                file=self.stream,
                flush=True,
            )

            self._write_log()
            self.plot_history()

        self.epoch += 1
        self._synced_losses["train"].reset()
        self._synced_losses["val"].reset()

    def plot_history(self, show: bool = False):
        """
        Plot history of current losses into self.plot_path.

        Arguments:
            show: Whether to show the plot on screen.
        """
        if self.global_rank > 0:
            return
        x = range(1, len(self.train_losses) + 1)
        n_rows, n_cols = _calc_plot_dim(len(self.loss_labels), f=0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.expand_dims(axes, axis=0)
        for i, (label, ax) in enumerate(zip(self.loss_labels, axes.flatten())):
            ax.semilogy(x, self.train_losses[:, i], "-bx")
            ax.semilogy(x, self.val_losses[:, i], "-gx")
            ax.legend(["Training", "Validation"])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            if self.loss_weights[i]:
                label = f"{label} (x {self.loss_weights[i]})"
            ax.set_title(label)
        fig.tight_layout()
        plt.savefig(self.plot_path)
        print(f"Loss history plot saved to {self.plot_path}", file=self.stream, flush=True)
        if show:
            plt.show()
        else:
            plt.close()

    def get_joinable(self, mode: str = "train"):
        """Return a joinable for uneven training/validation inputs.

        Arguments:
            mode: str. Choose 'train or 'val'.
        """
        if mode not in ["train", "val"]:
            raise ValueError(f"mode should be 'train' or 'val', but got `{mode}`")
        return self._synced_losses[mode]
