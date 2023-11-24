import torch
import numpy as np


def test_EqGraphLoss():
    from mlspm.losses import GraphLoss

    # fmt:off

    node_classes_ref = [
        torch.Tensor([0, 1, 1]).long(),
        torch.Tensor([1, 1, 0, 0]).long()
    ]
    edges_ref = [
        torch.Tensor([
            [0, 1],
            [1, 2]
        ]).long(),
        torch.Tensor([
            [0, 1, 1],
            [1, 2, 3]
        ]).long()
    ]
    node_classes_pred = [
        torch.Tensor([
            [5.0, 5.0],
            [4.0, 6.0],
            [7.0, 3.0]
        ]).log(),
        torch.Tensor([
            [2.0, 8.0],
            [9.0, 1.0],
            [9.0, 1.0],
            [6.0, 4.0]
        ]).log()
    ]
    edge_classes_pred = [
        torch.Tensor([0.9, 0.3, 0.8]),
        torch.Tensor([0.2, 0.9, 0.5, 0.7, 0.4])
    ]
    edges_pred = [
        torch.Tensor([
            [0, 0, 1],
            [1, 2, 2]
        ]).long(),
        torch.Tensor([
            [0, 0, 1, 1, 2],
            [1, 2, 2, 3, 3]
        ]).long()
    ]
    ref = (node_classes_ref, edges_ref)
    pred = (node_classes_pred, edge_classes_pred, edges_pred)

    criterion = GraphLoss(1, 2)
    losses = criterion(pred, ref, separate_loss_factors=True)

    assert torch.allclose(losses[0], torch.Tensor([2.3322996])), losses[0]
    assert torch.allclose(losses[1], torch.Tensor([0.7928372])), losses[1]
    assert torch.allclose(losses[2], torch.Tensor([0.7697312])), losses[2]

    # fmt: on
