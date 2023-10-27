from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_padding(kernel_size: int | Tuple[int, ...], nd: int) -> Tuple[int, ...]:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    padding = []
    for i in range(nd):
        padding += [(kernel_size[i] - 1) // 2]
    return tuple(padding)


class _ConvNdBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nd: int,
        kernel_size: int | Tuple[int, ...] = 3,
        depth: int = 2,
        padding_mode: str = "zeros",
        res_connection: bool = True,
        activation: bool = None,
        last_activation: bool = True,
    ):
        assert depth >= 1

        if nd == 2:
            conv = nn.Conv2d
        elif nd == 3:
            conv = nn.Conv3d
        else:
            raise ValueError(f"Invalid convolution dimensionality {nd}.")

        super().__init__()

        self.res_connection = res_connection
        if not activation:
            self.act = nn.ReLU()
        else:
            self.act = activation

        if last_activation:
            self.acts = [self.act] * depth
        else:
            self.acts = [self.act] * (depth - 1) + [self._identity]

        padding = _get_padding(kernel_size, nd)
        self.convs = nn.ModuleList(
            [conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)]
        )
        for _ in range(depth - 1):
            self.convs.append(conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode))
        if res_connection and in_channels != out_channels:
            self.res_conv = conv(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def _identity(self, x):
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x))
        if self.res_connection:
            if self.res_conv:
                x = x + self.res_conv(x_in)
            else:
                x = x + x_in
        return x


class Conv2dBlock(_ConvNdBlock):
    """
    Pytorch 2D convolution block module.

    Arguments:
        in_channels: Number of channels entering the first convolution layer.
        out_channels: Number of output channels in each layer of the block.
        kernel_size: Size of convolution kernel.
        depth: Number of convolution layers in the block.
        padding_mode: str. Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        res_connection: Whether to use residual connection over the block (f(x) = h(x) + x). If in_channels != out_channels,
            a 1x1x1 convolution is applied to the res connection to make the channel numbers match.
        activation: Activation function to use after every layer in block. If None, defaults to ReLU.
        last_activation: Whether to apply the activation after the last conv layer (before res connection).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int] = 3,
        depth: int = 2,
        padding_mode: int = "zeros",
        res_connection: bool = True,
        activation: Optional[nn.Module] = None,
        last_activation: bool = True,
    ):
        super().__init__(in_channels, out_channels, 2, kernel_size, depth, padding_mode, res_connection, activation, last_activation)


class Conv3dBlock(_ConvNdBlock):
    """
    Pytorch 3D convolution block module.

    Arguments:
        in_channels: Number of channels entering the first convolution layer.
        out_channels: Number of output channels in each layer of the block.
        kernel_size: Size of convolution kernel.
        depth: Number of convolution layers in the block.
        padding_mode: Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        res_connection: Whether to use residual connection over the block (f(x) = h(x) + x). If in_channels != out_channels,
            a 1x1x1 convolution is applied to the res connection to make the channel numbers match.
        activation: Activation function to use after every layer in block. If None, defaults to ReLU.
        last_activation: Whether to apply the activation after the last conv layer (before res connection).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        depth: int = 2,
        padding_mode: int = "zeros",
        res_connection: bool = True,
        activation: Optional[nn.Module] = None,
        last_activation: bool = True,
    ):
        super().__init__(in_channels, out_channels, 3, kernel_size, depth, padding_mode, res_connection, activation, last_activation)


class UNetAttentionConv(nn.Module):
    """
    Pytorch attention layer for U-net model upsampling stage.

    Arguments:
        in_channels: Number of channels in the attended feature map.
        query_channels: Number of channels in query feature map.
        attention_channels: Number of channels in hidden convolution layer before computing attention.
        kernel_size: Size of convolution kernel.
        padding_mode: Type of padding in each convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
        conv_activation: Activation function to use after convolution layers
        attention_activation: Type of activation to use for attention map. 'sigmoid' or 'softmax'.
        upsample_mode: Algorithm for upsampling query feature map to the attended feature map size. See torch.nn.functional.interpolate.
        ndim: 2 or 3. Dimensionality of convolution.

    References:
        https://arxiv.org/abs/1804.03999
    """

    def __init__(
        self,
        in_channels: int,
        query_channels: int,
        attention_channels: int,
        kernel_size: int | Tuple[int, ...],
        padding_mode: str = "zeros",
        conv_activation: nn.Module = nn.ReLU(),
        attention_activation: str = "softmax",
        upsample_mode: str = "bilinear",
        ndim: int = 2,
    ):
        super().__init__()

        self.ndim = ndim
        if ndim == 2:
            conv = nn.Conv2d
        elif ndim == 3:
            conv = nn.Conv3d
        else:
            raise ValueError(f"Invalid convolution dimensionality {ndim}.")

        if attention_activation == "softmax":
            self.attention_activation = self._softmax
        elif attention_activation == "sigmoid":
            self.attention_activation = self._sigmoid
        else:
            raise ValueError(f"Unrecognized attention map activation {attention_activation}.")

        padding = _get_padding(kernel_size, ndim)
        self.x_conv = conv(in_channels, attention_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        self.q_conv = conv(query_channels, attention_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        self.a_conv = conv(attention_channels, 1, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample_mode = upsample_mode
        self.conv_activation = conv_activation

    def _softmax(self, a):
        shape = a.shape
        return self.softmax(a.reshape(shape[0], -1)).reshape(shape)

    def _sigmoid(self, a):
        return self.sigmoid(a)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Upsample query q to the size of input x and convolve
        q = F.interpolate(q, size=x.size()[2:], mode=self.upsample_mode, align_corners=False)
        q = self.conv_activation(self.q_conv(q))

        # Convolve input x and sum with q
        a = self.conv_activation(self.x_conv(x))
        a = self.conv_activation(a + q)

        # Get attention map and mix it with x
        a = self.attention_activation(self.a_conv(a))
        x = a * x

        return x, a.squeeze(dim=1)


class AttentionConvZ(nn.Module):
    """
    Reduce and expand 3D feature map in z direction by an attention convolution.

    Arguments:
        in_channels: Number of channels in input.
        z_out: Size of z dimension in output feature map.
        kernel_size: Convolution kernel size.
        conv_depth: Convolution block depth.
        padding_mode: Type of padding in convolution layer. 'zeros', 'reflect', 'replicate' or 'circular'.
    """

    def __init__(
        self,
        in_channels: int,
        z_out: int,
        kernel_size: int | Tuple[int, int, int] = 3,
        conv_depth: int = 2,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv3dBlock(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    depth=conv_depth,
                    padding_mode=padding_mode,
                    res_connection=False,
                    last_activation=False,
                )
                for _ in range(z_out)
            ]
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = []

        for conv in self.convs:
            # Compute attention maps
            a = self.act(conv(x))

            # Multiply attention maps with original input
            x_ = a * x

            # Reduce z dimension
            x_ = x_.sum(dim=-1)

            xs.append(x_)

        # Create new z dimension with the list of outputs
        x = torch.stack(xs, dim=-1)

        return x
