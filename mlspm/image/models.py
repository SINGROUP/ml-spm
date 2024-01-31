from os import PathLike
from typing import List, Literal, Optional, Tuple

import torch
from torch import nn

from .._weights import download_weights
from ..modules import Conv2dBlock, Conv3dBlock, UNetAttentionConv, _get_padding


def _flatten_z_to_channels(x):
    return x.permute(0, 4, 1, 2, 3).reshape(x.size(0), -1, x.size(2), x.size(3))


class AttentionUNet(nn.Module):
    """
    Pytorch 3D-to-2D U-net model with attention.

    3D conv -> concatenate -> 3D conv/pool/dropout -> 2D conv/dropout -> 2D upsampling/conv with skip connections
    and attention. For multiple inputs, the inputs are first processed through separate 3D conv blocks before merging
    by concatenating along channel axis.

    Arguments:
        conv3d_in_channels: Number of channels in input.
        conv2d_in_channels: Number of channels in first 2D conv layer after flattening 3D to 2D.
        conv3d_out_channels: Number of channels after 3D-to-2D flattening after each 3D conv block. Depends on input z size.
        n_in: Number of input 3D images.
        n_out: Number of output 2D maps.
        merge_block_channels: Number of channels in input merging 3D conv blocks.
        merge_block_depth: Number of layers in each merge conv block.
        conv3d_block_channels: Number channels in 3D conv blocks.
        conv3d_block_depth: Number of layers in each 3D conv block.
        conv3d_dropouts: Dropout rates after each conv3d block.
        conv2d_block_channels: Number channels in 2D conv blocks.
        conv2d_block_depth: Number of layers in each 2D conv block.
        conv2d_dropouts: Dropout rates after each conv2d block.
        attention_channels: Number of channels in conv layer within each attention block.
        upscale2d_block_channels: Number of channels in each 2D conv block after upscale before skip connection.
        upscale2d_block_depth: Number of layers in each 2D conv block after upscale before skip connection.
        upscale2d_block_channels2: Number of channels in each 2D conv block after skip connection.
        upscale2d_block_depth2: Number of layers in each 2D conv block after skip connection.
        split_conv_block_channels: Number of channels in 2d conv blocks after splitting outputs.
        split_conv_block_depth: Number of layers in each 2d conv block after splitting outputs.
        res_connections: Whether to use residual connections in conv blocks.
        out_convs_channels: Number of channels in splitted outputs.
        out_relus: Whether to apply relu activation to the output 2D maps.
        pool_type: Type of pooling to use.
        pool_z_strides: Stride of pool layers in z direction.
        padding_mode: Type of padding in each convolution layer.
        activation: Activation to use after every layer except last one.
        attention_activation: Type of activation to use for attention map.
        device: Device to load model onto.
    """

    def __init__(
        self,
        conv3d_in_channels: int,
        conv2d_in_channels: int,
        conv3d_out_channels: List[int],
        n_in: int = 1,
        n_out: int = 3,
        merge_block_channels: List[int] = [8],
        merge_block_depth: int = 2,
        conv3d_block_channels: List[int] = [8, 16, 32],
        conv3d_block_depth: int = 2,
        conv3d_dropouts: List[float] = [0.0, 0.0, 0.0],
        conv2d_block_channels: List[int] = [128],
        conv2d_block_depth: int = 3,
        conv2d_dropouts: List[float] = [0.1],
        attention_channels: List[int] = [32, 32, 32],
        upscale2d_block_channels: List[int] = [16, 16, 16],
        upscale2d_block_depth: int = 1,
        upscale2d_block_channels2: List[int] = [16, 16, 16],
        upscale2d_block_depth2: int = 2,
        split_conv_block_channels: List[int] = [16],
        split_conv_block_depth: List[int] = [3],
        res_connections: bool = True,
        out_convs_channels: int | List[int] = 1,
        out_relus: bool | List[bool] = True,
        pool_type: Literal["avg", "max"] = "avg",
        pool_z_strides: List[int] = [2, 1, 2],
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        activation: Literal["relu", "lrelu", "elu"] | nn.Module = "lrelu",
        attention_activation: Literal["sigmoid", "softmax"] = "softmax",
        device: str = "cuda",
    ):
        super().__init__()

        assert (
            len(conv3d_block_channels)
            == len(conv3d_out_channels)
            == len(conv3d_dropouts)
            == len(upscale2d_block_channels)
            == len(upscale2d_block_channels2)
            == len(attention_channels)
        )

        if isinstance(activation, nn.Module):
            self.act = activation
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "lrelu":
            self.act = nn.LeakyReLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Unknown activation function {activation}")

        if not isinstance(out_relus, list):
            out_relus = [out_relus] * n_out
        else:
            assert len(out_relus) == n_out

        if not isinstance(out_convs_channels, list):
            out_convs_channels = [out_convs_channels] * n_out
        else:
            assert len(out_convs_channels) == n_out

        self.out_relus = out_relus
        self.relu_act = nn.ReLU()

        # -- Input merge conv blocks --
        self.merge_convs = nn.ModuleList([None] * n_in)
        for i in range(n_in):
            self.merge_convs[i] = nn.ModuleList(
                [
                    Conv3dBlock(
                        conv3d_in_channels,
                        merge_block_channels[0],
                        3,
                        merge_block_depth,
                        padding_mode,
                        res_connections,
                        self.act,
                        False,
                    )
                ]
            )
            for j in range(len(merge_block_channels) - 1):
                self.merge_convs[i].append(
                    Conv3dBlock(
                        merge_block_channels[j],
                        merge_block_channels[j + 1],
                        3,
                        merge_block_depth,
                        padding_mode,
                        res_connections,
                        self.act,
                        False,
                    )
                )

        # -- Encoder conv blocks --
        self.conv3d_blocks = nn.ModuleList(
            [
                Conv3dBlock(
                    n_in * merge_block_channels[-1],
                    conv3d_block_channels[0],
                    3,
                    conv3d_block_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
                )
            ]
        )
        self.conv3d_dropouts = nn.ModuleList([nn.Dropout(conv3d_dropouts[0])])
        for i in range(len(conv3d_block_channels) - 1):
            self.conv3d_blocks.append(
                Conv3dBlock(
                    conv3d_block_channels[i],
                    conv3d_block_channels[i + 1],
                    3,
                    conv3d_block_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
                )
            )
            self.conv3d_dropouts.append(nn.Dropout(conv3d_dropouts[i + 1]))

        # -- Middle conv blocks --
        self.conv2d_blocks = nn.ModuleList(
            [
                Conv2dBlock(
                    conv2d_in_channels, conv2d_block_channels[0], 3, conv2d_block_depth, padding_mode, res_connections, self.act, False
                )
            ]
        )
        self.conv2d_dropouts = nn.ModuleList([nn.Dropout(conv2d_dropouts[0])])
        for i in range(len(conv2d_block_channels) - 1):
            self.conv2d_blocks.append(
                Conv2dBlock(
                    conv2d_block_channels[i],
                    conv2d_block_channels[i + 1],
                    3,
                    conv2d_block_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
                )
            )
            self.conv2d_dropouts.append(nn.Dropout(conv2d_dropouts[i + 1]))

        # -- Decoder conv blocks --
        self.attentions = nn.ModuleList([])
        for c_att, c_conv in zip(attention_channels, reversed(conv3d_out_channels)):
            self.attentions.append(
                UNetAttentionConv(
                    c_conv, conv2d_block_channels[-1], c_att, 3, padding_mode, self.act, attention_activation, upsample_mode="bilinear"
                )
            )

        self.upscale2d_blocks = nn.ModuleList(
            [
                Conv2dBlock(
                    conv2d_block_channels[-1],
                    upscale2d_block_channels[0],
                    3,
                    upscale2d_block_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
                )
            ]
        )
        for i in range(len(upscale2d_block_channels) - 1):
            self.upscale2d_blocks.append(
                Conv2dBlock(
                    upscale2d_block_channels2[i],
                    upscale2d_block_channels[i + 1],
                    3,
                    upscale2d_block_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
                )
            )

        self.upscale2d_blocks2 = nn.ModuleList([])
        for i in range(len(upscale2d_block_channels2)):
            self.upscale2d_blocks2.append(
                Conv2dBlock(
                    upscale2d_block_channels[i] + conv3d_out_channels[-(i + 1)],
                    upscale2d_block_channels2[i],
                    3,
                    upscale2d_block_depth2,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
                )
            )

        # -- Output split conv blocks --
        padding = _get_padding(3, 2)
        self.out_convs = nn.ModuleList([])
        self.split_convs = nn.ModuleList([None] * n_out)
        for i_out in range(n_out):
            self.split_convs[i_out] = nn.ModuleList(
                [
                    Conv2dBlock(
                        upscale2d_block_channels2[-1],
                        split_conv_block_channels[0],
                        3,
                        split_conv_block_depth,
                        padding_mode,
                        res_connections,
                        self.act,
                        False,
                    )
                ]
            )
            for i in range(len(split_conv_block_channels) - 1):
                self.split_convs.append(
                    Conv2dBlock(
                        split_conv_block_channels[i],
                        split_conv_block_channels[i + 1],
                        3,
                        split_conv_block_depth,
                        padding_mode,
                        res_connections,
                        self.act,
                        False,
                    )
                )

            self.out_convs.append(
                nn.Conv2d(
                    split_conv_block_channels[-1], out_convs_channels[i_out], kernel_size=3, padding=padding, padding_mode=padding_mode
                )
            )

        if pool_type == "avg":
            pool = nn.AvgPool3d
        elif pool_type == "max":
            pool = nn.MaxPool3d
        self.pools = nn.ModuleList([pool(2, stride=(2, 2, pz)) for pz in pool_z_strides])

        self.upsample2d = nn.Upsample(scale_factor=2, mode="nearest")
        self.device = device
        self.n_out = n_out
        self.n_in = n_in

        self.to(device)

    def _flatten(self, x):
        return x.permute(0, 1, 4, 2, 3).reshape(x.size(0), -1, x.size(2), x.size(3))

    def forward(self, x: List[torch.Tensor]):
        """
        Do forward computation.

        Arguments:
            x: Input AFM images of shape (batch, channels, )
        """
        assert len(x) == self.n_in

        # Do 3D convolutions for each input
        in_branches = []
        for xi, convs in zip(x, self.merge_convs):
            for conv in convs:
                xi = self.act(conv(xi))
            in_branches.append(xi)

        # Merge input branches
        x = torch.cat(in_branches, dim=1)

        # Encode
        x_3ds = []
        for conv, dropout, pool in zip(self.conv3d_blocks, self.conv3d_dropouts, self.pools):
            x = self.act(conv(x))
            x = dropout(x)
            x_3ds.append(x)
            x = pool(x)

        # Middle 2d convs
        x = self._flatten(x)
        for conv, dropout in zip(self.conv2d_blocks, self.conv2d_dropouts):
            x = self.act(conv(x))
            x = dropout(x)

        # Compute attention maps
        attention_maps = []
        x_gated = []
        for attention, x_3d in zip(self.attentions, reversed(x_3ds)):
            g, a = attention(self._flatten(x_3d), x)
            x_gated.append(g)
            attention_maps.append(a)

        # Decode
        for i, (conv1, conv2, xg) in enumerate(zip(self.upscale2d_blocks, self.upscale2d_blocks2, x_gated)):
            x = self.upsample2d(x)
            x = self.act(conv1(x))
            x = torch.cat([x, xg], dim=1)  # Attention-gated skip connection
            x = self.act(conv2(x))

        # Split into different outputs
        outputs = []
        for i, (split_convs, out_conv) in enumerate(zip(self.split_convs, self.out_convs)):
            h = x
            for conv in split_convs:
                h = self.act(conv(h))
            h = out_conv(h)

            if self.out_relus[i]:
                h = self.relu_act(h)
            outputs.append(h.squeeze(1))

        return outputs, attention_maps


class EDAFMNet(AttentionUNet):
    """
    ED-AFM Attention U-net.

    This is the model used in the ED-AFM paper for task of predicting electrostatics from AFM images.
    It is a subclass of the :class:`AttentionUNet` class with specific hyperparameters.

    The following pretrained weights are available:

        - ``'base'``: The base model used for all predictions in the main ED-AFM paper and used for comparison in the various test in the supplementary information of the paper.
        - ``'single-channel'``: Model trained on only a single CO-tip AFM input.
        - ``'CO-Cl'``: Model trained on alternative tip combination of CO and Cl.
        - ``'Xe-Cl'``: Model trained on alternative tip combination of Xe and Cl.
        - ``'constant-noise'``: Model trained using constant noise amplitude instead of normally distributed amplitude.
        - ``'uniform-noise'``: Model trained using uniform random noise amplitude instead of normally distributed amplitude.
        - ``'no-gradient'``: Model trained without background-gradient augmentation.
        - ``'matched-tips'``: Model trained on data with matched tip distance between CO and Xe, instead of independently randomized distances.

    Arguments:
        device: Device to load model onto.
        trained_weights: If not None, load the specified pretrained weights to the model.
        weights_dir: If **weights_type** is not None, the directory where the weights will be downloaded into.
    """

    def __init__(
        self,
        device: str = "cuda",
        pretrained_weights: Optional[
            Literal["base", "single-channel", "CO-Cl", "Xe-Cl", "constant-noise", "uniform-noise", "no-gradient", "matched-tips"]
        ] = None,
        weights_dir: PathLike = "./weights",
    ):
        if pretrained_weights == "single-channel":
            n_in = 1
        else:
            n_in = 2

        super().__init__(
            conv3d_in_channels=1,
            conv2d_in_channels=192,
            merge_block_channels=[32],
            merge_block_depth=2,
            conv3d_out_channels=[288, 288, 384],
            conv3d_dropouts=[0.0, 0.0, 0.0],
            conv3d_block_channels=[48, 96, 192],
            conv3d_block_depth=3,
            conv2d_block_channels=[512],
            conv2d_block_depth=3,
            conv2d_dropouts=[0.0],
            n_in=n_in,
            n_out=1,
            upscale2d_block_channels=[256, 128, 64],
            upscale2d_block_depth=2,
            upscale2d_block_channels2=[256, 128, 64],
            upscale2d_block_depth2=2,
            split_conv_block_channels=[64],
            split_conv_block_depth=3,
            out_relus=[False],
            pool_z_strides=[2, 1, 2],
            activation=nn.LeakyReLU(negative_slope=0.1, inplace=True),
            padding_mode="replicate",
            device=device,
        )

        if pretrained_weights:
            weights_path = download_weights(f"ed-afm-{pretrained_weights}", weights_dir)
            self.load_state_dict(torch.load(weights_path))


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
        last_relu: bool | Tuple[bool, ...] = [False, True, True],
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
