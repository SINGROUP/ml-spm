import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import AttentionConvZ, Conv3dBlock, UNetAttentionConv, _get_padding
from . import Atom, MoleculeGraph, find_gaussian_peaks, make_box_borders
from .._weights import download_weights


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
        activation: Activation to use after every layer. ``'relu'``, ``'lrelu'``, or ``'elu'`` or :class:`nn.Module`.
        padding_mode: Type of padding in each convolution layer. ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        pool_type: Type of pooling to use. ``'avg'`` or ``'max'``.
        decoder_z_sizes: Upscale sizes of decoder stages in the z dimension.
        z_outs: Size of the z-dimension after encoder and skip connections.
        attention_activation: Type of activation to use for attention map. ``'sigmoid'`` or ``'softmax'``.
        afm_res: Real-space size of pixels in xy-plane in input AFM images in angstroms.
        grid_z_range: The real-space range in z-direction of the position grid in angstroms. Of the format ``(z_min, z_max)``.
        peak_std: Standard deviation of atom position grid peaks in angstroms.
        match_threshold: Detection threshold for matching when finding atom position peaks.
        match_method: Method for template matching when finding atom position peaks. See :func:`.find_gaussian_peaks` for options.
        device: Device to store model on.
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
        device: str = "cuda",
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
                Conv3dBlock(
                    encode_block_channels[i],
                    encode_block_channels[i + 1],
                    3,
                    encode_block_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                )
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
                    decode_block_channels2[i],
                    decode_block_channels[i],
                    3,
                    decode_block_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                    False,
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
        self.out_conv = nn.Conv3d(
            decode_block_channels2[-1],
            1,
            kernel_size=3,
            padding=_get_padding(3, 3),
            padding_mode=padding_mode,
        )

        # Attention convolution for dealing with variable z sizes at the end of the encoder
        enc_channels = self.encode_block_channels
        self.att_conv_enc = AttentionConvZ(enc_channels[-1], z_outs[0], conv_depth=3, padding_mode=self.padding_mode)

        # Attention convolutions for the skip connections
        self.att_conv_skip = nn.ModuleList(
            [
                AttentionConvZ(c, z_out, conv_depth=3, padding_mode=self.padding_mode)
                for c, z_out in zip(reversed(enc_channels[:-1]), z_outs[1:])
            ]
        )

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Arguments:
            x: Batch of AFM images. Should be of shape ``(n_batch, nx, ny, nz)```.

        Returns:
            Tuple (**pos_dist**, **attention_maps**), where

            - **pos_dist** - Predicted atom position distribution.
            - **attention_maps** - Attention maps from the skip connection attention layers.
        """
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

        return x, attention_maps

    def get_positions(
        self, x: torch.Tensor | np.ndarray, device: str = "cuda"
    ) -> Tuple[list[torch.Tensor], torch.Tensor | np.ndarray, list[torch.Tensor | np.ndarray]]:
        """
        Predict atom positions for a batch of AFM images.

        Arguments:
            x: Batch of AFM images. Should be of shape ``(n_batch, nx, ny, nz)``.
            device: Device used when **x** is an np.ndarray.

        Returns:
            atom_pos: Atom positions for each batch item.
            grid: Atom position grid from PosNet prediction. Type matches input AFM image type
            attention: Attention maps from skip-connection attention layers. Type matches input AFM image type.
        """

        if isinstance(x, np.ndarray):
            xt = torch.from_numpy(x).float().to(device)
        else:
            xt = x

        with torch.no_grad():
            xt, attention = self(xt.unsqueeze(1))

        box_borders = make_box_borders(x.shape[1:3], (self.afm_res, self.afm_res), z_range=self.grid_z_range)
        atom_pos, _, _ = find_gaussian_peaks(
            xt,
            box_borders,
            match_threshold=self.match_threshold,
            std=self.peak_std,
            method=self.match_method,
        )

        if isinstance(x, np.ndarray):
            attention = [a.cpu().numpy() for a in attention]
            xt = xt.cpu().numpy()

        return atom_pos, xt, attention


class GraphImgNet(nn.Module):
    """
    Image-to-graph model that constructs a molecule graph out of atom positions and an AFM image.

    Arguments:
        posnet: :class:`PosNet` for predicting atom positions from an AFM image. Required when training or doing inference without
            pre-defined atom positions.
        n_classes: Number of different classes for nodes.
        iters: Number of message passing iterations.
        node_feature_size: Number of hidden node features.
        message_size: Size of message vector.
        message_hidden_size: Size of hidden layers in message MLP.
        edge_cutoff: Cutoff radius in angstroms for edges between atoms within MPNN.
        afm_cutoff: Cutoff radius in angstroms for receptive regions around each atom.
        afm_res: Real-space size of pixels in xy-plane in input AFM images in angstroms.
        conv_channels: Number channels in 3D conv blocks encoding AFM image regions.
        conv_depth: Number of layers in each 3D conv block.
        node_out_hidden_size: Size of hidden layers in node classification MLP.
        edge_out_hidden_size: Size of hidden layers in edge classification MLP.
        res_connections: Whether to use residual connections in conv blocks.
        activation: Activation to use after every layer. ``'relu'``, ``'lrelu'``, or ``'elu'`` or :class:`nn.Module`.
        padding_mode: Type of padding in each convolution layer. ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        pool_type: Type of pooling to use. ``'avg'`` or ``'max'``.
        device: Device to store model on.
    """

    def __init__(
        self,
        n_classes: int,
        posnet: Optional[PosNet] = None,
        iters: int = 3,
        node_feature_size: int = 20,
        message_size: int = 20,
        message_hidden_size: int = 64,
        edge_cutoff: float = 3.0,
        afm_cutoff: float = 1.25,
        afm_res: float = 0.125,
        conv_channels: list[int] = [4, 8, 16],
        conv_depth: int = 2,
        node_out_hidden_size: int = 128,
        edge_out_hidden_size: int = 128,
        res_connections: bool = True,
        activation: str | nn.Module = "relu",
        padding_mode: str = "zeros",
        pool_type: str = "avg",
        device: str = "cuda",
    ):
        super().__init__()

        self.posnet = posnet
        self.n_classes = n_classes
        self.iters = iters
        self.node_feature_size = node_feature_size
        self.message_size = message_size
        self.act = _get_activation(activation)
        self.edge_cutoff = edge_cutoff
        self.afm_cutoff = afm_cutoff
        self.afm_res = afm_res
        self.conv_channels = conv_channels
        self.padding_mode = padding_mode

        if (self.posnet is not None) and not np.allclose(self.posnet.afm_res, self.afm_res):
            warnings.warn(
                f"AFM pixel resolution ({self.afm_res}) does not match with the resolution in PosNet ({self.posnet.afm_res}). "
                "This can lead to bad inference results."
            )

        self.pool_type = pool_type
        pool = _get_pool(self.pool_type)
        self.pool = pool((2, 2, 1), stride=(2, 2, 1))  # Don't pool in z direction

        self.msg_net = nn.Sequential(
            nn.Linear(2 * node_feature_size + 3, message_hidden_size),
            self.act,
            nn.Linear(message_hidden_size, message_hidden_size),
            self.act,
            nn.Linear(message_hidden_size, message_size),
        )
        self.gru_node = nn.GRUCell(message_size, node_feature_size)
        self.gru_edge = nn.GRUCell(message_size, node_feature_size)

        conv_in_channels = [1] + conv_channels[:-1]
        self.conv_blocks = nn.ModuleList(
            [
                Conv3dBlock(
                    conv_in_channels[i],
                    conv_channels[i],
                    3,
                    conv_depth,
                    padding_mode,
                    res_connections,
                    self.act,
                )
                for i in range(len(self.conv_channels))
            ]
        )

        # Attention convolution for dealing with variable feature maps size at the end of the AFM image encoder
        in_channels = self.conv_channels[-1]
        self.att_conv = Conv3dBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            depth=3,
            padding_mode=self.padding_mode,
            res_connection=False,
            last_activation=False,
        )

        self.node_transform = nn.Linear(conv_channels[-1], node_feature_size)

        self.class_net = nn.Sequential(
            nn.Linear(node_feature_size, node_out_hidden_size),
            self.act,
            nn.Linear(node_out_hidden_size, n_classes),
        )
        self.edge_net = nn.Sequential(
            nn.Linear(node_feature_size, edge_out_hidden_size),
            self.act,
            nn.Linear(edge_out_hidden_size, 1),
            nn.Sigmoid(),
        )

        self.device = device
        self.to(device)

    def _gather_afm(self, x: torch.Tensor, pos: list[torch.Tensor]) -> torch.Tensor:
        if sum([len(p) for p in pos]) == 0:
            # No atom positions, so just return an empty tensor
            print("Encountered an empty position list")
            return torch.zeros((0, 1, 1, 1), device=self.device)

        ind_radius = int(self.afm_cutoff / self.afm_res)

        # Pad AFM image so that image regions on the edges are the same size
        x = F.pad(
            x,
            (0, 0, ind_radius, ind_radius, ind_radius, ind_radius),
            mode="constant",
            value=0,
        )

        x_afm = []
        for ib, p in enumerate(pos):
            # Find xy index range around each atom
            ind_min = ((p[:, :2] - self.afm_cutoff) / self.afm_res).round().long()
            ind_min += ind_radius  # Add radius due to padding
            ind_max = ind_min + 2 * ind_radius + 1

            for ia in range(p.shape[0]):
                x_afm.append(
                    x[
                        ib,
                        ind_min[ia, 0] : ind_max[ia, 0],
                        ind_min[ia, 1] : ind_max[ia, 1],
                    ]
                )

        return torch.stack(x_afm, axis=0)

    def _get_edges(self, pos: list[torch.Tensor]) -> list[torch.Tensor]:
        edges = []
        for p in pos:
            d = F.pdist(p)
            inds = torch.nonzero(d <= self.edge_cutoff)[:, 0]
            edges.append(torch.triu_indices(len(p), len(p), offset=1, device=self.device)[:, inds])
        return edges

    def _combine_graphs(self, pos: list[torch.Tensor], edges: list[torch.Tensor]):
        Ns = []
        ind_count = 0
        edges_shifted = []
        for p, e in zip(pos, edges):
            edges_shifted += [[e_[0] + ind_count, e_[1] + ind_count] for e_ in e.T]
            ind_count += len(p)
            Ns.append(len(p))
        edges_shifted = torch.tensor(edges_shifted, device=self.device, dtype=torch.long).T
        pos = torch.cat(pos, axis=0)
        return edges_shifted, pos, Ns

    def pred_to_graph(
        self,
        pos: list[torch.Tensor],
        node_classes: list[torch.Tensor],
        edge_classes: list[torch.Tensor],
        edges: list[torch.Tensor],
        bond_threshold: float,
    ) -> list[MoleculeGraph]:
        """
        Convert predicted batch to a simple list of molecule graphs.

        Arguments:
            pos: Atom positions for each batch item.
            node_classes: Predicted class probabilities for each atom in the molecule graphs. Each batch item is a tensor
                of shape ``(n_atoms, n_classes)``.
            edge_classes: Predicted probabilities for the existence of bonds between atoms indicated by **edges**. Eatch batch
                item is a tensor of shape ``(n_edges,)``.
            edges: Possible bond connection indices between atoms. Each batch item is a tensor of shape ``(2, n_edges)``.
            bond_threshold: Threshold probability when an edge is considered a bond between atoms.

        Returns:
            Molecule graphs corresponding to the predictions.
        """
        graphs = []
        for p, nc, e, ec in zip(pos, node_classes, edges, edge_classes):
            nc = F.softmax(nc, dim=1)
            atoms = [Atom(pi.cpu().numpy(), class_weights=nci.cpu().numpy()) for pi, nci in zip(p, nc)]
            et = e[:, ec >= bond_threshold].cpu().numpy()
            bonds = [tuple(b) for b in et.T]
            graphs.append(MoleculeGraph(atoms, bonds))
        return graphs

    def encode_afm(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.zeros((0, self.node_feature_size), device=self.device)

        # Apply convolutions
        x = x.unsqueeze(1)
        for conv in self.conv_blocks:
            x = self.act(conv(x))
            x = self.pool(x)

        # Reduce xyz dimensions to 1 by attention convolution
        sh = x.shape
        a = self.att_conv(x)  # Get attention maps
        a = F.softmax(a.reshape(sh[0], sh[1], -1), dim=2).reshape(sh)  # Softmax so that xyz dimension sum to 1 for every channel
        x = (a * x).sum(dim=(2, 3, 4))  # Multiply by attention and reduce over xyz

        # Transform features by a linear layer
        x = self.node_transform(self.act(x))

        return x

    def mpnn(self, pos: torch.Tensor, node_features: torch.Tensor, edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Ne = 0 if edges.ndim < 2 else edges.size(1)  # Number of edges

        if Ne > 0:
            # Symmetrise directional edge connections
            edges_sym = torch.cat([edges, edges[[1, 0]]], dim=1)

            # Compute vectors between nodes connected by edges
            src_pos = pos.index_select(0, edges_sym[0])
            dst_pos = pos.index_select(0, edges_sym[1])
            d_pos = dst_pos - src_pos

            # Initialize edge features to the average of the nodes they are connecting
            src_features = node_features.index_select(0, edges[0])
            dst_features = node_features.index_select(0, edges[1])
            edge_features = (src_features + dst_features) / 2

        else:
            edge_features = torch.empty((0, self.node_feature_size), device=self.device)

        for _ in range(self.iters):
            a = torch.zeros(node_features.size(0), self.message_size, device=self.device)
            if Ne > 0:  # No messages if no edges
                # Gather start and end nodes of edges
                src_features = node_features.index_select(0, edges_sym[0])
                dst_features = node_features.index_select(0, edges_sym[1])
                inputs = torch.cat([src_features, dst_features, d_pos], dim=1)

                # Calculate messages for all edges and add them to start nodes
                messages = self.msg_net(inputs)
                a.index_add_(0, edges_sym[0], messages)

                # Update edge features
                b = (messages[:Ne] + messages[Ne:]) / 2  # Average over two directions
                edge_features = self.gru_edge(b, edge_features)

            # Update node features
            node_features = self.gru_node(a, node_features)

        return node_features, edge_features

    def forward(
        self, x: torch.Tensor, pos: Optional[list[torch.Tensor]] = None
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Arguments:
            x: Batch of AFM images. Array of shape ``(n_batch, nx, ny, nz)``.
            pos: Atom positions for each batch item. If None, the positions are predicted from the AFM images.
                The positions should be such that the lower left corner of the AFM image is at coordinate ``(0, 0)``,
                and all positions are within the bounds of the AFM image. Model needs to be constructed with PosNet
                defined in order for the position prediction to work.

        Returns:
            Tuple (**node_classes**, **edge_classes**, **edges**), where

            - **node_classes** - Predicted class probabilities for each atom in the molecule graphs. Each batch item is a tensor
              of shape ``(n_atoms, n_classes)``.
            - **edge_classes** - Predicted probabilities for the existence of bonds between atoms indicated by **edges**. Eatch batch
              item is a tensor of shape ``(n_edges,)``.
            - **edges** - Possible bond connection indices between atoms. Each batch item is a tensor of shape ``(2, n_edges)``.
        """

        assert x.ndim == 4, "Wrong number of dimensions in AFM tensor. Should be 4."

        if pos is None:
            if self.posnet is None:
                raise RuntimeError(f"Attempting to predict atom positions, but PosNet is not defined.")
            pos, _, _ = self.posnet.get_positions(x)

        # Gather all AFM image regions to one tensor
        x = self._gather_afm(x, pos)

        # Get AFM embeddings for each node
        x_afm = self.encode_afm(x)

        # Propagate features with MPNN
        edges = self._get_edges(pos)  # Get edges based on distances between atoms
        edges_combined, pos_combined, Ns = self._combine_graphs(pos, edges)  # Combine graphs into one for faster processing
        node_features, edge_features = self.mpnn(pos_combined, x_afm, edges_combined)

        # Predict node and edge classes
        node_classes = self.class_net(node_features)
        edge_classes = self.edge_net(edge_features).squeeze(1)

        # Split into batch of separate graphs again
        node_classes = torch.split(node_classes, split_size_or_sections=Ns)
        edge_classes = torch.split(edge_classes, split_size_or_sections=[e.size(1) for e in edges])

        return node_classes, edge_classes, edges

    def predict_graph(
        self,
        x: torch.Tensor | np.ndarray,
        pos: Optional[torch.Tensor] = None,
        bond_threshold: float = 0.5,
    ) -> Tuple[list[MoleculeGraph], Optional[torch.Tensor | np.ndarray]]:
        """
        Predict molecule graphs from AFM images.

        Arguments:
            X: Batch of AFM images. Array of shape ``(n_batch, nx, ny, nz)``.
            pos: Atom positions for each batch item. If None, the positions are predicted from the AFM images.
                The positions should be such that the lower left corner of the AFM image is at coordinate ``(0, 0)``,
                and all positions are within the bounds of the AFM image.
            bond_threshold: Threshold probability when an edge is considered a bond between atoms.

        Returns:
            Tuple (**graphs**, **grid**), where

            - **graphs**: Predicted graphs.
            - **grid**: Atom position grid from PosNet prediction when input **pos** is ``None``.
              Type matches input AFM image type.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        if pos is None:
            if self.posnet is None:
                raise RuntimeError(f"Attempting to predict positions, but PosNet is not defined.")
            pos, grid, _ = self.posnet.get_positions(x)
        else:
            grid = None
        node_classes, edge_classes, edges = self.forward(x, pos)
        graphs = self.pred_to_graph(pos, node_classes, edge_classes, edges, bond_threshold)
        return graphs, grid


class GraphImgNetIce(GraphImgNet):
    """
    GraphImgNet with hyperparameters set exactly as in the paper "Structure discovery in Atomic Force Microscopy imaging of ice",
    https://arxiv.org/abs/2310.17161.

    Three sets of pretrained weights are available:

        - ``'cu111'``: trained on images of ice clusters on Cu(111)
        - ``'au111-monolayer'``: trained on images of ice clusters on monolayer Au(111)
        - ``'au111-bilayer'``: trained on images of ice clusters on bilayer Au(111)

    Arguments:
        pretrained_weights: Name of pretrained weights. If specified, load pretrained weights. Otherwise, weights are initialized
            randomly.
        grid_z_range: The real-space range in z-direction of the position grid in angstroms. Of the format ``(z_min, z_max)``.
            Has to be specified when **pretrained_weights** is not given.
        device: Device to store model on.
    """

    def __init__(self, pretrained_weights: Optional[str] = None, grid_z_range: Optional[Tuple[float, float]] = None, device="cuda"):
        if pretrained_weights is not None:
            ice_z_ranges = {
                "cu111": (-2.9, 0.5),
                "au111-monolayer": (-2.9, 0.5),
                "au111-bilayer": (-3.5, 0.5),
            }
            z_range_weights = ice_z_ranges[pretrained_weights]
            if (grid_z_range is not None) and not np.allclose(z_range_weights, grid_z_range):
                warnings.warn(f"Specified grid z range ({grid_z_range}) does not match one for pretrained_weights ({z_range_weights})")
            else:
                grid_z_range = z_range_weights
        elif grid_z_range is None:
            raise ValueError("At least one of pretrained_weights or grid_z_range has to be specified.")

        outsize = round((grid_z_range[1] - grid_z_range[0]) / 0.1) + 1

        posnet = PosNet(
            encode_block_channels=[16, 32, 64, 128],
            encode_block_depth=3,
            decode_block_channels=[128, 64, 32],
            decode_block_depth=2,
            decode_block_channels2=[128, 64, 32],
            decode_block_depth2=3,
            attention_channels=[128, 128, 128],
            res_connections=True,
            activation="relu",
            padding_mode="zeros",
            pool_type="avg",
            decoder_z_sizes=[5, 15, outsize],
            z_outs=[3, 3, 5, 10],
            attention_activation="softmax",
            afm_res=0.125,
            grid_z_range=grid_z_range,
            peak_std=0.20,
            match_threshold=0.7,
            match_method="msd_norm",
            device=device,
        )
        super().__init__(
            n_classes=2,
            posnet=posnet,
            iters=5,
            node_feature_size=40,
            message_size=40,
            message_hidden_size=196,
            edge_cutoff=3.0,
            afm_cutoff=1.125,
            afm_res=0.125,
            conv_channels=[12, 24, 48],
            conv_depth=2,
            node_out_hidden_size=196,
            edge_out_hidden_size=196,
            res_connections=True,
            activation="relu",
            padding_mode="zeros",
            pool_type="avg",
            device=device,
        )
        if pretrained_weights is not None:
            weights_name = f"graph-ice-{pretrained_weights}"
            weights_path = download_weights(weights_name)
            weights = torch.load(weights_path)
            self.load_state_dict(weights)
