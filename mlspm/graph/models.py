from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import AttentionConvZ, Conv3dBlock, UNetAttentionConv, _get_padding
from .utils import find_gaussian_peaks


def _get_activation(activation):
    if isinstance(activation, nn.Module):
        return activation
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "lrelu":
        return nn.LeakyReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation function {activation}")


def _get_pool(pool_type):
    if pool_type == "avg":
        return nn.AvgPool3d
    elif pool_type == "max":
        return nn.MaxPool3d
    else:
        raise ValueError(f"Unknown pool type {pool_type}")


class PosNet(nn.Module):
    """
    Attention U-net for predicting the positions of atoms in an AFM image.

    Arguments:
        encode_block_channels: Number channels in encoding 3D conv blocks.
        encode_block_depth: Number of layers in each encoding 3D conv block.
        decode_block_channels: Number of channels in each decoding 3D conv block after upscale before skip connection.
        decode_block_depth: Number of layers in each decoding 3D conv block after upscale before skip connection.
        decode_block_channels2: Number of channels in each decoding 3D conv block after skip connection.
        decode_block_depth2: Number of layers in each decoding 3D conv block after skip connection.
        attention_channels: Number of channels in conv layers within each attention block.
        res_connections: Whether to use residual connections in conv blocks.
        activation: Activation to use after every layer. 'relu', 'lrelu', or 'elu' or nn.Module.
        padding_mode: Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        decoder_z_sizes: Upscale sizes of decoder stages in the z dimension.
        z_outs: Size of the z-dimension after encoder and skip connections.
        attention_activation: Type of activation to use for attention map. 'sigmoid' or 'softmax'.
        afm_res: Real-space size of pixels in xy-plane in input AFM images in angstroms.
        grid_z_range: The real-space range in z-direction of the position grid in angstroms. Of the format (z_min, z_max).
        peak_std: Standard deviation of atom position grid peaks in angstroms.
        match_threshold: Detection threshold for matching when finding atom position peaks.
        match_method: Method for template matching when finding atom position peaks. See .utils.find_gaussian_peaks for options.
    """

    def __init__(
        self,
        encode_block_channels: list[int] = [4, 8, 16, 32],
        encode_block_depth: int = 2,
        decode_block_channels: list[int] = [32, 16, 8],
        decode_block_depth: int = 2,
        decode_block_channels2: list[int] = [32, 16, 8],
        decode_block_depth2: int = 2,
        attention_channels: list[int] = [32, 32, 32],
        res_connections: bool = True,
        activation: str = "relu",
        padding_mode: str = "replicate",
        pool_type: str = "avg",
        decoder_z_sizes: list[int] = [5, 10, 20],
        z_outs: list[int] = [3, 3, 5, 10],
        attention_activation: str = "softmax",
        afm_res: float = 0.125,
        grid_z_range: Tuple[float, float] = (-1.4, 0.5),
        peak_std: float = 0.3,
        match_threshold: float = 0.7,
        match_method: str = "msd_norm",
    ):
        super().__init__()

        assert (
            len(encode_block_channels) == len(decoder_z_sizes) + 1 == len(decode_block_channels) + 1 == len(decode_block_channels2) + 1
        ), "Numbers of blocks do not match"

        self.encode_block_channels = encode_block_channels
        self.num_blocks = len(encode_block_channels)
        self.act = _get_activation(activation)
        self.decoder_z_sizes = decoder_z_sizes
        self.upsample_mode = "trilinear"
        self.padding_mode = padding_mode
        self.afm_res = afm_res
        self.grid_z_range = grid_z_range
        self.peak_std = peak_std
        self.match_threshold = match_threshold
        self.match_method = match_method

        self.pool_type = pool_type
        pool = _get_pool(self.pool_type)
        self.pool = pool((2, 2, 1), stride=(2, 2, 1))  # No pool in z-dimension

        # Encoder conv blocks
        encode_block_channels = [1] + encode_block_channels
        self.encode_blocks = nn.ModuleList(
            [
                Conv3dBlock(encode_block_channels[i], encode_block_channels[i + 1], 3, encode_block_depth, padding_mode, res_connections, self.act)
                for i in range(self.num_blocks)
            ]
        )

        # Skip-connection attention conv blocks
        self.unet_attentions = nn.ModuleList(
            [
                UNetAttentionConv(
                    encode_block_channels[-(i + 2)],
                    encode_block_channels[-1],
                    attention_channels[i],
                    3,
                    padding_mode,
                    self.act,
                    attention_activation,
                    upsample_mode=self.upsample_mode,
                    ndim=3,
                )
                for i in range(self.num_blocks - 1)
            ]
        )

        # Decoder conv blocks
        decode_block_channels2 = [encode_block_channels[-1]] + decode_block_channels2
        self.decode_blocks = nn.ModuleList(
            [
                Conv3dBlock(
                    decode_block_channels2[i], decode_block_channels[i], 3, decode_block_depth, padding_mode, res_connections, self.act, False
                )
                for i in range(self.num_blocks - 1)
            ]
        )
        self.decode_blocks2 = nn.ModuleList(
            [
                Conv3dBlock(
                    decode_block_channels[i] + encode_block_channels[-(i + 2)],
                    decode_block_channels2[i + 1],
                    3,
                    decode_block_depth2,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
                )
                for i in range(self.num_blocks - 1)
            ]
        )
        self.out_conv = nn.Conv3d(decode_block_channels2[-1], 1, kernel_size=3, padding=_get_padding(3, 3), padding_mode=padding_mode)

        # Attention convolution for dealing with variable z sizes at the end of the encoder
        enc_channels = self.encode_block_channels
        self.att_conv_enc = AttentionConvZ(enc_channels[-1], z_outs[0], conv_depth=3, padding_mode=self.padding_mode)

        # Attention convolutions for the skip connections
        self.att_conv_skip = nn.ModuleList(
            [AttentionConvZ(c, z_out, conv_depth=3, padding_mode=self.padding_mode) for c, z_out in zip(reversed(enc_channels[:-1]), z_outs[1:])]
        )

    def make_box_borders(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Make grid box borders for a given grid xy shape.

        Arguments:
            shape: Grid xy shape.

        Returns: Box start and end coordinates in the form ((x_start, y_start, z_start), (x_end, y_end, z_end)).
        """
        # fmt:off
        box_borders = np.array(
            (                            0,                             0, self.grid_z_range[0]),
            (self.afm_res * (shape[0] - 1), self.afm_res * (shape[1] - 1), self.grid_z_range[1])
        )  # fmt:on
        return box_borders

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, list[torch.Tensor]]:
        xs = []
        for i in range(self.num_blocks):
            # Apply 3D conv block
            x = self.act(self.encode_blocks[i](x))

            if i < self.num_blocks - 1:
                # Store feature maps for attention gating later
                xs.append(x)

                # Down-sample for next iteration of convolutions.
                x = self.pool(x)

        # Apply attention convolution to get to a fixed z size
        x = self.att_conv_enc(x)

        # Compute skip-connection attention maps
        attention_maps = []
        x_gated = []
        xs.reverse()
        for attention, att_z, x_ in zip(self.unet_attentions, self.att_conv_skip, xs):
            xg, a = attention(x_, x)
            xg = att_z(xg)
            x_gated.append(xg)
            attention_maps.append(a)

        # Decode
        for i, (conv1, conv2, xg) in enumerate(zip(self.decode_blocks, self.decode_blocks2, x_gated)):
            # Upsample and apply first conv block
            target_size = xs[i].shape[2:4] + (self.decoder_z_sizes[i],)
            x = F.interpolate(x, size=target_size, mode=self.upsample_mode, align_corners=False)
            x = self.act(conv1(x))

            # Concatenate attention-gated skip connections and apply second conv block
            xg = F.interpolate(xg, size=target_size, mode=self.upsample_mode, align_corners=False)
            x = torch.cat([x, xg], dim=1)
            x = self.act(conv2(x))

        # Get output grid
        x = self.out_conv(x).squeeze(1)

        if return_attention:
            x = (x, attention_maps)

        return x

    def get_positions(
        self, x: torch.Tensor | np.ndarray, device: str = "cuda"
    ) -> Tuple[list[torch.Tensor], torch.Tensor | np.ndarray, list[torch.Tensor | np.ndarray]]:
        """
        Predict atom positions for a batch of AFM images.

        Arguments:
            x: Batch of AFM images. Should be of shape (nb, nx, ny, nz).
            device: Device used when x is an np.ndarray.

        Returns:
            atom_pos: Atom positions for each batch item.
            grid: Atom position grid from PosNet prediction. Type matches input AFM image type
            attention: Attention maps from attention layers. Type matches input AFM image type.
        """

        if isinstance(x, np.ndarray):
            xt = torch.from_numpy(x).float().to(device)
        else:
            xt = x

        with torch.no_grad():
            xt, attention = self.forward(xt.unsqueeze(1), return_attention=True)

        box_borders = self.make_box_borders(x.shape[1:3])
        atom_pos, _, _ = find_gaussian_peaks(xt, box_borders, match_threshold=self.match_threshold, std=self.peak_std, method=self.match_method)

        if isinstance(x, np.ndarray):
            attention = [a.cpu().numpy() for a in attention]
            xt = xt.cpu().numpy()

        return atom_pos, xt, attention
