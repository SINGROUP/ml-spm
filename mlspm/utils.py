import glob
import os
import re
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

elements = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
]


def _calc_plot_dim(n: int, f: float = 0.3):
    rows = max(int(np.sqrt(n) - f), 1)
    cols = 1
    while rows * cols < n:
        cols += 1
    return rows, cols


def _get_distributed() -> Tuple[int, int, int, Optional[dist.ProcessGroup]]:
    try:
        if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            world_size = dist.get_world_size()
            local_rank = global_rank = dist.get_rank()
        group = dist.group.WORLD
    except (RuntimeError, AssertionError, ValueError):
        world_size = 1
        group = None
    if world_size <= 1:
        world_size = 1
        local_rank = global_rank = 0
    return world_size, local_rank, global_rank, group


def _print_progress(block_num: int, block_size: int, total_size: int):
    if total_size == -1:
        return
    delta = block_size / total_size * 100
    current_size = block_num * block_size
    percent = current_size / total_size * 100
    percent_int = int(percent)
    if (percent - percent_int) > 1.0001 * delta:
        # Only print when crossing an integer percentage
        return
    if block_num > 0:
        print("\b\b\b", end="", flush=True)
    if current_size < total_size:
        print(f"{percent_int:2d}%", end="", flush=True)
    else:
        print("Done")


class Checkpointer:
    """
    Keep checkpoints of a Pytorch model and optimizer over epochs, keeping only the one
    with best loss. Also load the latest checkpoint in the beginning if any exist.

    Arguments:
        model: Pytorch model.
        optimizer: Pytorch optimizer.
        additional_module: Additional modules whose state will be saved to the checkpoints.
        checkpoint_dir: Path to directory where checkpoints are saved.
        keep_last_epoch: Also keep the last epoch even if it does not have the best loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        additional_data: dict = {},
        checkpoint_dir: str = "./Checkpoints",
        keep_last_epoch: bool = True,
    ):
        if hasattr(model, "module"):
            model = model.module
        self.model = model
        self.optimizer = optimizer
        self.additional_data = additional_data
        self.additional_data["best_loss"] = np.inf
        self.additional_data["best_epoch"] = 0
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_epoch = keep_last_epoch
        self.world_size, self.local_rank, self.global_rank, self.group = _get_distributed()
        self._get_init_epoch()

    @property
    def best_epoch(self):
        return self.additional_data["best_epoch"]

    @property
    def best_loss(self):
        return self.additional_data["best_loss"]

    def _get_init_epoch(self):
        dir_exists = os.path.exists(self.checkpoint_dir)
        if self.world_size > 1:
            dist.barrier()
        if not dir_exists:
            if self.global_rank == 0:
                os.makedirs(self.checkpoint_dir)
        cp_files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pth"))
        if len(cp_files) == 0:
            self.epoch = 1
            return
        self.epoch = sorted([int(re.search("[0-9]+", os.path.split(p)[1]).group(0)) for p in cp_files])[-1]
        last_cp_path = os.path.join(self.checkpoint_dir, f"model_{self.epoch}.pth")
        load_checkpoint(self.model, self.optimizer, last_cp_path, self.additional_data, self.local_rank)
        self.epoch += 1

    def next_epoch(self, loss: float):
        """
        Advance epoch and save state if loss improved.

        Arguments:
            loss: Loss value for the current epoch.
        """
        if self.global_rank == 0:
            improved = loss < self.best_loss
            prev_best_epoch = self.best_epoch
            if improved:
                if prev_best_epoch > 0:
                    os.remove(os.path.join(self.checkpoint_dir, f"model_{prev_best_epoch}.pth"))
                self.additional_data["best_loss"] = loss
                self.additional_data["best_epoch"] = self.epoch
            if self.keep_last_epoch and prev_best_epoch != (self.epoch - 1):
                os.remove(os.path.join(self.checkpoint_dir, f"model_{self.epoch - 1}.pth"))
            if self.keep_last_epoch or improved:
                save_checkpoint(self.model, self.optimizer, self.epoch, self.checkpoint_dir, additional_data=self.additional_data)
            if self.world_size > 1:
                dist.broadcast(torch.tensor(self.additional_data["best_loss"], device=self.local_rank, dtype=torch.float), src=0)
                dist.broadcast(torch.tensor(self.additional_data["best_epoch"], device=self.local_rank, dtype=torch.long), src=0)
        else:
            best_loss = torch.tensor(0, device=self.local_rank, dtype=torch.float)
            best_epoch = torch.tensor(0, device=self.local_rank, dtype=torch.long)
            dist.broadcast(best_loss, src=0)
            dist.broadcast(best_epoch, src=0)
            self.additional_data["best_loss"] = best_loss.cpu().item()
            self.additional_data["best_epoch"] = best_epoch.cpu().item()
        self.epoch += 1

    def revert_to_best_epoch(self):
        """Revert model state to the best epoch."""
        best_cp_path = os.path.join(self.checkpoint_dir, f'model_{self.additional_data["best_epoch"]}.pth')
        load_checkpoint(self.model, self.optimizer, best_cp_path, self.additional_data)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, save_dir: str, additional_data: dict = {}):
    """
    Save a Pytorch model/optimizer checkpoint.

    Arguments:
        model: Model whose state to save.
        optimizer: Optimizer whose state to save.
        epoch: Training epoch.
        save_dir: Directory to save in.
        additional_data: A dictionary of additional modules or data to save to the checkpoint.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if hasattr(model, "module"):
        model = model.module

    state = {
        "model_params": model.state_dict(),
        "optim_params": optimizer.state_dict(),
    }
    for key in additional_data:
        data = additional_data[key]
        if hasattr(data, "state_dict"):
            data = data.state_dict()
        state[key] = data

    save_path = os.path.join(save_dir, f"model_{epoch}.pth")
    torch.save(state, save_path)
    print(f"Model, optimizer weights on epoch {epoch} saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    file_name: str = "./model.pth",
    additional_data: Optional[dict] = None,
    rank: Optional[int] = None,
):
    """
    Load a Pytorch model/optimizer checkpoint.

    Arguments:
        model: Model whose state to load.
        optimizer: Optimizer whose state to load.
        file_name: Checkpoint file to load from.
        additional_data: If not None, a dictionary with additional modules or other
            data that will be updated with additional data found in the checkpoint.
        rank: Process rank for distributed training.
    """
    import torch

    if rank is None:
        state = torch.load(file_name, weights_only=False)
    else:
        state = torch.load(file_name, map_location={"cuda:0": f"cuda:{rank}"}, weights_only=False)
    model.load_state_dict(state["model_params"])

    if optimizer:
        optimizer.load_state_dict(state["optim_params"])
        print(f"Model, optimizer weights loaded from {file_name}")
    else:
        print(f"Model weights loaded from {file_name}")

    if additional_data is not None:
        for key in state:
            if key in additional_data:
                data = additional_data[key]
                if hasattr(data, "load_state_dict"):
                    data.load_state_dict(state[key])
                    print(f"Loaded state for `{key}`")
                else:
                    additional_data[key] = state[key]
                    print(f"Updated data for `{key}`")


def read_xyzs(file_paths: list[str], return_comment: bool = False) -> list[np.ndarray]:
    """
    Read molecule xyz files.

    Arguments:
        file_paths: Paths to xyz files
        return_comment: If True, also return the comment string on second line of file.

    Returns:
        Arrays of shape ``(num_atoms, 4)`` or ``(num_atoms, 5)``. Each row in the arrays corresponds to one atom
        with ``[x, y, z, element]`` or ``[x, y, z, charge, element]``.
    """
    mols = []
    comments = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            N = int(f.readline().strip())
            comments.append(f.readline())
            atoms = []
            for line in f:
                line = line.strip().split()
                try:
                    elem = int(line[0])
                except ValueError:
                    elem = elements.index(line[0]) + 1
                posc = [float(p) for p in line[1:]]
                atoms.append(posc + [elem])
        mols.append(np.array(atoms))
    if return_comment:
        mols = mols, comments
    return mols


def write_to_xyz(molecule: np.ndarray, outfile: str = "./pos.xyz", comment_str: str = "", verbose: int = 1):
    """
    Write molecule into xyz file.

    Arguments:
        molecule: Molecule to write. np.array of shape (num_atoms, 4) or (num_atoms, 5).
            Each row corresponds to one atom with [x, y, z, element] or [x, y, z, charge, element].
        outfile: Path where xyz file will be saved.
        comment_str: Comment written to the second line of the file.
        verbose: 0 or 1. Whether to print output information.
    """
    molecule = molecule[molecule[:, -1] > 0]
    with open(outfile, "w") as f:
        f.write(f"{len(molecule)}\n{comment_str}\n")
        for atom in molecule:
            f.write(f"{int(atom[-1])}\t")
            for i in range(len(atom) - 1):
                f.write(f"{atom[i]:10.8f}\t")
            f.write("\n")
    if verbose > 0:
        print(f"Molecule xyz file saved to {outfile}")


def batch_write_xyzs(xyzs: list[np.ndarray], outdir: str = "./", start_ind: int = 0, verbose: int = 1):
    """
    Write a batch of xyz files 0_mol.xyz, 1_mol.xyz, ...

    Arguments:
        xyzs: Molecules to write.
        outdir: Directory where files are saved.
        start_ind: Index where file numbering starts.
        verbose: 0 or 1. Whether to print output information.
    """
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    ind = start_ind
    for xyz in xyzs:
        write_to_xyz(xyz, os.path.join(outdir, f"{ind}_mol.xyz"), verbose=verbose)
        ind += 1


def count_parameters(module: torch.nn.Module) -> int:
    """
    Count trainable parameters in a Pytorch module.

    Arguments:
        module: Pytorch module.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
