import torch
import numpy as np

torch.set_printoptions(precision=9)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def test_PosNet():
    from mlspm.models import PosNet

    cnn = PosNet(
        encode_block_channels=[4, 8, 16, 32],
        encode_block_depth=2,
        decode_block_channels=[24, 16, 6],
        decode_block_depth=2,
        decode_block_channels2=[64, 32, 8],
        decode_block_depth2=2,
        attention_channels=[32, 24, 16],
        res_connections=True,
        activation="relu",
        padding_mode="zeros",
        decoder_z_sizes=[4, 8, 16],
        attention_activation="softmax",
        device=device,
    )

    for i in range(1, 11):
        # All input sizes in z should result in same output size in z
        X = torch.rand(2, 1, 128, 96, i).to(device)
        x, attention_maps = cnn(X)

        assert x.shape == (2, 128, 96, 16)
        assert len(attention_maps) == 3
        # fmt:off
        assert (
                attention_maps[0].shape == (2,  32, 24, i)
            and attention_maps[1].shape == (2,  64, 48, i)
            and attention_maps[2].shape == (2, 128, 96, i)
        )
        # fmt:on


def test_GraphImgNet():
    from mlspm.models import GraphImgNet, PosNet

    posnet = PosNet(afm_res=0.2, peak_std=0.25, device=device)
    model = GraphImgNet(
        n_classes=4,
        posnet=posnet,
        node_feature_size=10,
        message_size=5,
        message_hidden_size=16,
        edge_cutoff=1.0,
        afm_cutoff=0.4,
        afm_res=0.2,
        node_out_hidden_size=32,
        edge_out_hidden_size=24,
        device=device,
    )

    # fmt:off

    # Test afm region selection
    X = torch.stack(
        [
            torch.linspace(0, 79, 80).reshape(8, 10).T,
            torch.linspace(0, 79, 80).reshape(10, 8)
        ],
        axis=0
    ).unsqueeze(3).to(device)
    atom_pos = [
        torch.Tensor([
            [0.40, 0.60, 0],
            [0.11, 0.20, 0]
        ]).to(device),
        torch.Tensor([
            [1.80, 1.40, 0]
        ]).to(device)
    ]
    x_afm = model._gather_afm(X, atom_pos)
    assert x_afm.shape == (3, 5, 5, 1)
    assert torch.allclose(x_afm, torch.Tensor([
        [[10, 20, 30, 40, 50],
         [11, 21, 31, 41, 51],
         [12, 22, 32, 42, 52],
         [13, 23, 33, 43, 53],
         [14, 24, 34, 44, 54]],
        [[ 0,  0,  0,  0,  0],
         [ 0,  0, 10, 20, 30],
         [ 0,  1, 11, 21, 31],
         [ 0,  2, 12, 22, 32],
         [ 0,  3, 13, 23, 33]],
        [[61, 62, 63,  0,  0],
         [69, 70, 71,  0,  0],
         [77, 78, 79,  0,  0],
         [ 0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0]]
    ]).unsqueeze(3).to(device))

    # Test AFM encoder network
    for k in range(1, 11):
        x_afm = model.encode_afm(torch.rand(3, 20, 20, k).to(device))
        assert x_afm.shape == (3, 10)

    # Test edge finding
    atom_pos = [
        torch.Tensor([
            [1.0, 1.0, 0.0],
            [1.0, 1.5, 0.0],
            [1.5, 1.0, 0.0],
            [1.9, 1.5, 0.0]
        ]).to(device),
        torch.empty(0, 3).to(device),
        torch.Tensor([
            [1.0, 1.0, -1.0],
            [1.0, 1.5, -1.0],
            [1.5, 1.5,  0.0],
            [2.0, 2.0,  0.0],
            [0.0, 0.0, -0.5]
        ]).to(device)
    ]
    edges = model._get_edges(atom_pos)
    assert len(edges) == 3
    assert torch.allclose(edges[0], torch.Tensor([
        [0, 0, 1, 1, 2],
        [1, 2, 2, 3, 3]
    ]).long().to(device)), edges[0]
    assert edges[1].shape == (2, 0), edges[1]
    assert torch.allclose(edges[2], torch.Tensor([
        [0, 2],
        [1, 3]
    ]).long().to(device)), edges[2]

    # Test graph combination
    edges_combined, pos, Ns = model._combine_graphs(atom_pos, edges)
    assert edges_combined.shape == (2, 7), edges_combined.shape
    assert torch.allclose(edges_combined, torch.Tensor([
        [0, 0, 1, 1, 2, 4, 6],
        [1, 2, 2, 3, 3, 5, 7]
    ]).long().to(device)), edges_combined
    assert pos.shape == (9, 3), pos.shape
    assert torch.allclose(pos, torch.Tensor([
        [1.0, 1.0,  0.0],
        [1.0, 1.5,  0.0],
        [1.5, 1.0,  0.0],
        [1.9, 1.5,  0.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.5, -1.0],
        [1.5, 1.5,  0.0],
        [2.0, 2.0,  0.0],
        [0.0, 0.0, -0.5]
    ]).to(device)), pos
    assert Ns == [4, 0, 5], Ns

    # Test MPNN
    x_afm = torch.rand(9, 10).to(device)
    node_features, edge_features = model.mpnn(pos, x_afm, edges_combined)
    assert node_features.shape == (9, 10), node_features.shape
    assert edge_features.shape == (7, 10), edge_features.shape

    # Test translational invariance of the MPNN
    pos += torch.Tensor([1.0, 1.0, 1.0]).to(device)
    node_features_shifted, edge_features_shifted = model.mpnn(pos, x_afm, edges_combined)
    assert torch.allclose(node_features, node_features_shifted, rtol=1e-4, atol=1e-6)
    assert torch.allclose(edge_features, edge_features_shifted, rtol=1e-4, atol=1e-6)

    # Test that the edges are not directional
    node_features_reverse, edge_features_reverse = model.mpnn(pos, x_afm, edges_combined[[1, 0]])
    assert torch.allclose(node_features, node_features_reverse, rtol=1e-4, atol=1e-6)
    assert torch.allclose(edge_features, edge_features_reverse, rtol=1e-4, atol=1e-6)

    # Test whole model
    model.afm_cutoff = 0.8
    X = torch.rand(3, 20, 20, 10).to(device)
    node_classes, edge_classes, edges_ = model(X, atom_pos)
    assert len(node_classes) == 3
    assert node_classes[0].shape == (4, 4)
    assert node_classes[1].shape == (0, 4)
    assert node_classes[2].shape == (5, 4)
    assert len(edge_classes) == 3
    assert edge_classes[0].shape == (5,)
    assert edge_classes[1].shape == (0,)
    assert edge_classes[2].shape == (2,)
    assert len(edges_) == 3
    assert torch.allclose(edges[0], edges_[0]), edges_[0]
    assert torch.allclose(edges[1], edges_[1]), edges_[1]
    assert torch.allclose(edges[2], edges_[2]), edges_[2]

    # Test permutation equivariance
    perm0 = [1, 2, 3, 0]
    perm2 = [3, 1, 2, 4, 0]
    atom_pos_ = [p.clone() for p in atom_pos]
    atom_pos_[0] = atom_pos_[0][perm0]
    atom_pos_[2] = atom_pos_[2][perm2]
    node_classes_perm, _, _ = model(X, atom_pos_)
    assert torch.allclose(node_classes_perm[0], node_classes[0][perm0])
    assert torch.allclose(node_classes_perm[1], node_classes[1])
    assert torch.allclose(node_classes_perm[2], node_classes[2][perm2])

    # Test pred-to-graph conversion
    node_classes = [
        torch.Tensor([
            [1, 4, 3, 2],
            [5, 1, 1, 3],
            [7, 1, 1, 1],
            [2, 1, 3, 4],
        ]).to(device).log(),
        torch.empty((0, 4)).to(device),
        torch.Tensor([
            [1, 2, 1, 6],
            [2, 2, 1, 5],
            [2, 2, 2, 4],
            [1, 7, 1, 1],
            [2, 5, 2, 1]
        ]).to(device).log()
    ]
    edge_classes = [
        torch.Tensor([0.2, 0.8, 0.4, 0.6, 0.9]),
        torch.empty(0),
        torch.Tensor([0.4, 0.8])
    ]
    graphs = model.pred_to_graph(atom_pos, node_classes, edge_classes, edges, 0.5)
    assert len(graphs) == 3
    arr0, arr1, arr2 = [g.array(xyz=True, class_index=True, class_weights=True) for g in graphs]
    assert np.allclose(arr0,
        np.array([
            [1.0, 1.0, 0.0, 1, 0.1, 0.4, 0.3, 0.2],
            [1.0, 1.5, 0.0, 0, 0.5, 0.1, 0.1, 0.3],
            [1.5, 1.0, 0.0, 0, 0.7, 0.1, 0.1, 0.1],
            [1.9, 1.5, 0.0, 3, 0.2, 0.1, 0.3, 0.4]
        ])
    )
    assert len(arr1) == 0
    assert np.allclose(arr2,
        np.array([
            [1.0, 1.0, -1.0, 3, 0.1, 0.2, 0.1, 0.6],
            [1.0, 1.5, -1.0, 3, 0.2, 0.2, 0.1, 0.5],
            [1.5, 1.5,  0.0, 3, 0.2, 0.2, 0.2, 0.4],
            [2.0, 2.0,  0.0, 1, 0.1, 0.7, 0.1, 0.1],
            [0.0, 0.0, -0.5, 1, 0.2, 0.5, 0.2, 0.1]
        ])
    )
    assert np.allclose(np.array(graphs[0].bonds).T, np.array([
        [0, 1, 2],
        [2, 3, 3]
    ]))
    assert graphs[1].bonds == []
    assert np.allclose(np.array(graphs[2].bonds).T, np.array([
        [2],
        [3]
    ]))

    # fmt:on

def test_ASDAFMNet():

    import torch
    from mlspm.image.models import ASDAFMNet

    torch.manual_seed(0)

    model = ASDAFMNet(n_out=3, last_relu=[False, True, True])

    x = torch.rand((5, 1, 128, 128, 10))
    ys = model(x)

    assert len(ys) == 3
    assert ys[0].shape == ys[1].shape == ys[2].shape == (5, 128, 128)
    assert ys[1].min() >= 0.0
    assert ys[2].min() >= 0.0
