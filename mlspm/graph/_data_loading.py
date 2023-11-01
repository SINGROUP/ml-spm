import warnings
from typing import Tuple

import numpy as np
import torch

from . import MoleculeGraph


def collate_graph(
    batch: Tuple[np.ndarray, list[MoleculeGraph], list[np.ndarray]]
) -> Tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[MoleculeGraph], list[np.ndarray]]:
    """
    Collate graph samples into a batch.

    Arguments:
        batch:
            Tuple (**X**, **mols**, **xyz**), where

            - **X** - Input AFM image. Array of shape ``(batch_size, x, y, z)``.
            - **mols** - Input molecules.
            - **xyz** - List of original molecules. Arrays of shape ``(n_atoms, 5)``.

    Returns:
        Tuple (**X**, **pos**, **node_classes**, **edges**, **mols**, **xyz**), where
        
        - **X** - Input AFM images.
        - **pos** - Graph node xyz coordinates.
        - **node_classes** - Graph node class indices.
        - **edges** - Graph edge indices.
        - **mols** - Input molecules. Unchanged from input argument.
        - **xyz** - List of original molecules. Unchanged from input argument.
    """
    X, mols, xyz = batch
    pos = []
    node_classes = []
    edges = []
    remove_inds = []
    for i, m in enumerate(mols):
        if len(m) == 0:
            warnings.warn("empty molecule in batch", RuntimeWarning)
            remove_inds.append(i)
            continue
        m_array = m.array(xyz=True, class_index=True)
        pos.append(torch.from_numpy(m_array[:, :3]).float())
        node_classes.append(torch.from_numpy(m_array[:, 3]).long())
        edges.append(torch.Tensor(m.bonds).T.long())
    X = np.delete(X, remove_inds, axis=0)
    X = torch.from_numpy(X).float()
    return X, pos, node_classes, edges, mols, xyz
