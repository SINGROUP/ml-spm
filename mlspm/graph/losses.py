from typing import Tuple
import torch
from torch import nn


class GraphLoss(nn.Module):
    """
    Loss that compares two graphs.

    Arguments:
        node_factor: Weight for node classification loss.
        edge_factor: Weight for edge classification loss.
    """

    def __init__(self, node_factor: float = 1.0, edge_factor: float = 1.0):
        super().__init__()
        self.node_factor = node_factor
        self.edge_factor = edge_factor
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        pred: Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        ref: Tuple[list[torch.Tensor], list[torch.Tensor]],
        separate_loss_factors=False,
    ) -> torch.Tensor | list[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            pred: Predicted graph batch as returned by :meth:`.GraphImgNet.forward`
            ref: Reference graph batch. A tuple (**node_classes**, **edges**), where

                - **node_classes** - Node classes as class index numbers. List of tensors of shape ``(n_atoms,)``.
                - **edges** - Edges as pairs of node indices. List of tensors of shape ``(2, n_edges)``.
            separate_loss_factors: Whether to return a single total loss or a separated list of values with each loss component.

        Returns:
            Computed loss value. Either a single value when ``separate_loss_factors==False``, or a list ``[total_loss,
            node_loss, edge_loss]`` when ``separate_loss_factors==True``.
        """
        node_classes_pred, edge_classes_pred, edges_pred = pred
        node_classes_ref, edges_ref = ref

        assert len(node_classes_pred) == len(node_classes_ref) == len(edge_classes_pred) == len(edges_pred) == len(edges_ref)

        # Node classification loss
        node_classes_pred = torch.cat(node_classes_pred)
        node_classes_ref = torch.cat(node_classes_ref)
        node_loss = self.cross_entropy(node_classes_pred, node_classes_ref)

        # Loop over batch items
        edge_losses = []
        for ecp, ep, er in zip(edge_classes_pred, edges_pred, edges_ref):
            # Construct reference edge classes based on which atoms are actually bonded. er contains
            # a list of the actual bonds and ep is a list of potential bond connections matching the
            # list of predicted probabilities in ecp. Compute the distance of between all pairs in ep
            # and all pairs er, and note that the distance is zero when there is a match. If there
            # are no edges in ref, then skip.
            if er.size(0) > 0:
                ecr = (torch.cdist(ep.T.float(), er.T.float(), p=2).min(dim=1).values < 1e-3).float()
                edge_losses.append(self.bce(ecp, ecr))

        if len(edge_losses) > 0:
            edge_loss = torch.cat(edge_losses).mean()
        else:
            edge_loss = torch.tensor(0)

        loss = self.node_factor * node_loss + self.edge_factor * edge_loss

        if separate_loss_factors:
            loss = [loss, node_loss, edge_loss]

        return loss
