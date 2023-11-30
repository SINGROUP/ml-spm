from typing import Tuple

from torch import nn

from ..modules import _get_padding


class ASDAFMNet(nn.Module):
    def __init__(
        self,
        n_out: int = 3,
        activation: nn.Module = nn.LeakyReLU(0.1),
        padding_mode: str = "reflect",
        last_relu: bool | Tuple[bool, ...] = True,
    ):
        super().__init__()

        if not isinstance(last_relu, list):
            last_relu = [last_relu] * n_out
        self.last_relu = last_relu

        padding_3d = _get_padding(3, nd=3)
        padding_2d = _get_padding(3, nd=2)
        pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
            pool,
            nn.Conv3d(4, 8, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
            pool,
            nn.Conv3d(8, 16, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
            pool,
        )

        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=padding_3d, padding_mode=padding_mode),
            activation,
        )

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
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
