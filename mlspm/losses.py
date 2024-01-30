from typing import List
import torch
from torch import nn


from .graph.losses import GraphLoss


class WeightedMSELoss(nn.Module):
    """
    Weighted multi-component mean squared error loss.

    Arguments:
        loss_factors: Multiplicative weight factors for each loss component.
    """

    def __init__(self, loss_factors: List[float]):
        super().__init__()
        self.loss_factors = loss_factors

    def forward(self, pred: List[torch.Tensor], ref: List[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            pred: Prediction
            ref: Reference (ground truth)

        Returns:
            Computed losses. The first element is the total weighted loss, and the subsequent values are the unweighted
            losses for each component.
        """
        assert len(pred) == len(ref) == len(self.loss_factors)

        losses = []
        total_loss = 0.0
        for p, r, f in zip(pred, ref, self.loss_factors):
            loss = torch.mean((p - r) ** 2)
            total_loss += f * loss
            losses.append(loss.reshape(1))

        losses = torch.cat([total_loss.reshape(1)] + losses)

        return losses
