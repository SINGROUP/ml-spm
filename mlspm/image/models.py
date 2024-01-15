from turtle import forward
from typing import Literal, Optional, Tuple

import torch
from torch import nn

from ..modules import _get_padding
from .._weights import download_weights


def _flatten_z_to_channels(x):
    return x.permute(0, 4, 1, 2, 3).reshape(x.size(0), -1, x.size(2), x.size(3))


class ASDAFMNet(nn.Module):
    """
    The model used in the paper "Automated structure discovery in atomic force microscopy": https://doi.org/10.1126/sciadv.aay6913.

    Two different sets of pretrained weights are available:

        - ``'asdafm-light'``: trained on a set of molecules containing the elements H, C, N, O, and F.
        - ``'asdafm-heavy'``: trained on a set of molecules additionally containing the elements Si, P, S, Cl, and Br.

    Arguments:
        n_out: number of output branches.
        activation: Activation function used after each layer.
        padding_mode: Type of padding in each convolution layer. ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        last_relu: Whether to use a ReLU layer after the last convolution in each output branch. Either provide a single value
            that applies to all output branches or a list of values corresponding to each output branch.
        pretrained_weights: Name of pretrained weights. If specified, load pretrained weights. Otherwise, weights are initialized
            randomly. If loading the weights, the other arguments should be left in their default values, so that the model
            hyperparameters match the training.
    """

    def __init__(
        self,
        n_out: int = 3,
        activation: nn.Module = nn.LeakyReLU(0.1),
        padding_mode: str = "replicate",
        last_relu: bool | Tuple[bool, ...] = True,
        pretrained_weights: Optional[Literal["asdafm-light", "asdafm-heavy"]] = None,
    ):
        super().__init__()

        if not isinstance(last_relu, list):
            last_relu = [last_relu] * n_out
        elif len(last_relu) != n_out:
            raise ValueError(f"Length of last_relu ({len(last_relu)}) does not match with n_out ({n_out})")
        self.last_relu = last_relu

        padding_3d = _get_padding(3, nd=3)
        padding_2d = _get_padding(3, nd=2)

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(4, 8, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
            nn.AvgPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
        )

        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
            activation,
        )

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(64, 16, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
                    activation,
                    nn.Conv2d(16, 16, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
                    activation,
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(16, 16, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
                    activation,
                    nn.Conv2d(16, 16, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
                    activation,
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(16, 16, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
                    activation,
                    nn.Conv2d(16, 16, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
                    activation,
                    nn.Conv2d(16, 1, kernel_size=3, padding=padding_2d, padding_mode=padding_mode),
                )
                for _ in range(n_out)
            ]
        )

        if pretrained_weights is not None:
            weights_path = download_weights(pretrained_weights)
            weights = torch.load(weights_path)
            self.load_state_dict(weights)

    def forward(self, x):
        x = self.encoder(x)
        x = _flatten_z_to_channels(x)
        x = self.middle(x)
        ys = []
        for decoder, relu in zip(self.decoders, self.last_relu):
            y = decoder(x)
            if relu:
                y = nn.functional.relu(y)
            y = y.squeeze(1)
            ys.append(y)
        return ys
