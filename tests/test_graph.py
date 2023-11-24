import shutil
import torch
import pytest
import numpy as np


def test_collate_graph():
    from mlspm.graph import MoleculeGraph
    from mlspm.data_loading import collate_graph

    # fmt: off

    X_in = np.random.rand(4, 128, 128, 10)
    atoms = [
        np.array([
            [0.1, 0.2, 0.3, 1],
            [0.5, 0.6, 0.7, 1],
            [0.9, 1.0, 1.1, 0],
            [1.3, 1.4, 1.5, 0]
        ]),
        np.array([
            [1.7, 1.8, 1.9, 1],
            [2.1, 2.2, 2.3, 0]
        ]),
        np.empty((0, 4)),
        np.array([
            [2.5, 2.6, 2.7, 1],
            [2.9, 3.0, 3.1, 0],
            [3.3, 3.4, 3.5, 0]
        ]),
    ]
    bonds = [
        [(0,1), (0,2), (1,3), (2,3)],
        [(0,1)],
        [],
        [(0,1), (1,2)]
    ]
    classes = [[0], [1]]
    mols = [MoleculeGraph(a, b, classes) for a, b in zip(atoms, bonds)]
    xyz_in = np.random.rand(5, 3)

    with pytest.warns(RuntimeWarning):
        X, pos, node_classes, edges, ref_graphs, xyz = collate_graph((X_in, mols, xyz_in))

    assert torch.allclose(X, torch.from_numpy(X_in[[0, 1, 3]]).float())
    assert len(pos) == 3
    assert torch.allclose(pos[0], torch.Tensor([
        [0.1, 0.2, 0.3],
        [0.5, 0.6, 0.7],
        [0.9, 1.0, 1.1],
        [1.3, 1.4, 1.5]
    ]).float())
    assert torch.allclose(pos[1], torch.Tensor([
        [1.7, 1.8, 1.9],
        [2.1, 2.2, 2.3]
    ]).float())
    assert torch.allclose(pos[2], torch.Tensor([
        [2.5, 2.6, 2.7],
        [2.9, 3.0, 3.1],
        [3.3, 3.4, 3.5]
    ]).float())
    assert len(node_classes) == 3
    assert torch.allclose(node_classes[0], torch.Tensor([1, 1, 0, 0]).long())
    assert torch.allclose(node_classes[1], torch.Tensor([1, 0]).long())
    assert torch.allclose(node_classes[2], torch.Tensor([1, 0, 0]).long())
    assert len(edges) == 3
    assert torch.allclose(edges[0], torch.Tensor([
        [0, 0, 1, 2],
        [1, 2, 3, 3]
    ]).long())
    assert torch.allclose(edges[1], torch.Tensor([[0], [1]]).long())
    assert torch.allclose(edges[2], torch.Tensor([[0, 1], [1, 2]]).long())
    assert ref_graphs is mols
    assert xyz is xyz_in

    # fmt: on


def test_threshold_atoms():
    from mlspm.graph import MoleculeGraph, threshold_atoms_bonds

    # fmt: off
    atoms = [
        np.array([
            [0.1, 0.2,  0.3,  1],
            [0.3, 0.4, -2.0, 35],
            [0.5, 0.6, -1.2, 35],
            [0.7, 0.8, -0.4,  6]
        ])
    ]
    # fmt: on
    bonds = [[(0, 1), (0, 3), (1, 2), (2, 3)]]
    molecules = [MoleculeGraph(a, b) for a, b in zip(atoms, bonds)]

    new_molecules = threshold_atoms_bonds(molecules, threshold=-1.0, use_vdW=False)

    # fmt: off
    atoms_expected = [
        np.array([
            [0.1, 0.2,  0.3, 1],
            [0.7, 0.8, -0.4, 6]
        ])
    ]
    # fmt: on
    bonds_expected = [[(0, 1)]]
    molecules_expected = [MoleculeGraph(a, b) for a, b in zip(atoms_expected, bonds_expected)]

    assert len(new_molecules) == len(molecules_expected)
    for a, b in zip(new_molecules, molecules_expected):
        assert np.allclose(a.array(xyz=True, element=True), b.array(xyz=True, element=True))
        assert len(a.bonds) == len(b.bonds)
        for t1, t2 in zip(a.bonds, b.bonds):
            assert t1 == t2

    new_molecules = threshold_atoms_bonds(molecules, threshold=-1.0, use_vdW=True)

    # fmt: off
    atoms_expected = [
        np.array([
            [0.1, 0.2,  0.3,  1],
            [0.5, 0.6, -1.2, 35],
            [0.7, 0.8, -0.4,  6]
        ])
    ]
    # fmt: on
    bonds_expected = [[(0, 2), (1, 2)]]
    molecules_expected = [MoleculeGraph(a, b) for a, b in zip(atoms_expected, bonds_expected)]

    assert len(new_molecules) == len(molecules_expected)
    for a, b in zip(new_molecules, molecules_expected):
        assert np.allclose(a.array(xyz=True, element=True), b.array(xyz=True, element=True))
        assert len(a.bonds) == len(b.bonds)
        for t1, t2 in zip(a.bonds, b.bonds):
            assert t1 == t2


def test_find_bonds():
    from mlspm.graph import find_bonds

    # fmt: off
    molecules = [
        np.array([
            [0.0, 0.0,  0.0,  1],
            [1.0, 1.0,  0.0,  1],
            [1.0, 0.0,  0.0,  6],
            [2.0, 0.0, -1.0, 35],
        ])
    ]
    # fmt: on

    bonds = find_bonds(molecules)

    bonds_expected = [[(0, 2), (1, 2), (2, 3)]]

    assert len(bonds) == 1
    assert len(bonds[0]) == 3
    for a, b in zip(bonds[0], bonds_expected[0]):
        assert a == b

    print("All tests passed on find_bonds")


def test_molecule_graph_array():
    from mlspm.graph import MoleculeGraph

    # fmt: off
    
    atoms = np.array([
            [0.0, 0.0,  0.0, 1],
            [1.0, 1.0,  0.0, 2],
            [1.0, 0.0,  0.0, 3],
            [2.0, 0.0, -1.0, 4]
    ])
    bonds = [(0,2), (1,2), (2,3)]
    classes = [[1, 2], [3, 4]]
    class_weights = [
        [1.0, 0.0],
        [0.3, 0.7],
        [0.0, 1.0],
        [0.6, 0.4]
    ]
    molecule = MoleculeGraph(atoms, bonds, classes=classes, class_weights=None)

    atoms_array = molecule.array(xyz=True, q=False, element=False, class_weights=True)
    atoms_expected = np.array([
        [0.0, 0.0,  0.0, 1, 0],
        [1.0, 1.0,  0.0, 1, 0],
        [1.0, 0.0,  0.0, 0, 1],
        [2.0, 0.0, -1.0, 0, 1]
    ])
    assert np.allclose(atoms_array, atoms_expected)

    atoms_array = molecule.array(xyz=True, q=True, element=True, class_weights=True)
    atoms_expected = np.array([
        [0.0, 0.0,  0.0, 0, 1, 1, 0],
        [1.0, 1.0,  0.0, 0, 2, 1, 0],
        [1.0, 0.0,  0.0, 0, 3, 0, 1],
        [2.0, 0.0, -1.0, 0, 4, 0, 1]
    ])
    assert np.allclose(atoms_array, atoms_expected)

    molecule = MoleculeGraph(atoms, bonds, classes=None, class_weights=class_weights)
    atoms_array = molecule.array(xyz=True, class_index=True, class_weights=True)
    atoms_expected = np.array([
        [0.0, 0.0,  0.0, 0, 1.0, 0.0],
        [1.0, 1.0,  0.0, 1, 0.3, 0.7],
        [1.0, 0.0,  0.0, 1, 0.0, 1.0],
        [2.0, 0.0, -1.0, 0, 0.6, 0.4]
    ])
    assert np.allclose(atoms_array, atoms_expected)

    with pytest.raises(AssertionError):
        molecule = MoleculeGraph(atoms, bonds, classes=classes, class_weights=class_weights)

    # fmt: on


def test_molecule_graph_remove_atoms():
    from mlspm.graph import MoleculeGraph, Atom

    # fmt: off

    atoms = np.array([
            [0.0, 0.0,  0.0, 1],
            [1.0, 1.0,  0.0, 2],
            [1.0, 0.0,  0.0, 3],
            [2.0, 0.0, -1.0, 4]
    ])
    bonds = [(0,2), (1,2), (2,3)]
    molecule = MoleculeGraph(atoms, bonds)

    for i in range(4):
        new_molecule, removed = molecule.remove_atoms(list(range(i)))
        assert len(removed) == i
        assert len(new_molecule.atoms) == 4-i
        assert np.allclose(molecule.array(xyz=True, element=True), atoms)
        assert len(molecule.bonds) == 3

    new_molecule, removed = molecule.remove_atoms([])
    assert np.allclose(molecule.array(xyz=True, element=True), new_molecule.array(xyz=True, element=True))
    assert len(new_molecule.bonds) == 3
    for a, b in zip(bonds, new_molecule.bonds):
        assert a == b
    assert removed == []

    new_molecule, removed = molecule.remove_atoms([1])
    removed_expected = [(Atom(np.array([1.0, 1.0, 0.0]), 2), [0, 1, 0])]
    atoms_expected = np.array([
        [0.0, 0.0,  0.0, 1],
        [1.0, 0.0,  0.0, 3],
        [2.0, 0.0, -1.0, 4]
    ])
    bonds_expected = [(0,1), (1,2)]
    for a, b in zip(removed, removed_expected):
        assert np.allclose(a[0].array(), b[0].array())
        assert np.allclose(a[1], b[1])
    assert np.allclose(new_molecule.array(xyz=True, element=True), atoms_expected)
    for a, b in zip(new_molecule.bonds, bonds_expected):
        assert a == b

    new_molecule, removed = molecule.remove_atoms([1,3])
    removed_expected = [
        (Atom(np.array([1.0, 1.0,  0.0]), 2), [0, 1]),
        (Atom(np.array([2.0, 0.0, -1.0]), 4), [0, 1])
    ]
    atoms_expected = np.array([
        [0.0, 0.0, 0.0, 1],
        [1.0, 0.0, 0.0, 3]
    ])
    bonds_expected = [(0,1)]
    for a, b in zip(removed, removed_expected):
        assert np.allclose(a[0].array(xyz=True, element=True), b[0].array(xyz=True, element=True))
        assert np.allclose(a[1], b[1])
    assert np.allclose(new_molecule.array(xyz=True, element=True), atoms_expected)
    for a, b in zip(new_molecule.bonds, bonds_expected):
        assert a == b

    new_molecule, removed = molecule.remove_atoms([0,1,2,3])
    removed_expected = [
        (Atom(np.array([0.0, 0.0,  0.0]), 1), []),
        (Atom(np.array([1.0, 1.0,  0.0]), 2), []),
        (Atom(np.array([1.0, 0.0,  0.0]), 3), []),
        (Atom(np.array([2.0, 0.0, -1.0]), 4), [])
    ]
    for a, b in zip(removed, removed_expected):
        assert np.allclose(a[0].array(xyz=True, element=True), b[0].array(xyz=True, element=True))
        assert np.allclose(a[1], b[1])
    assert new_molecule.array(xyz=True, element=True) == []
    assert new_molecule.bonds == []

    # fmt: on


def test_molecule_graph_add_atom():
    from mlspm.graph import Atom, MoleculeGraph

    # fmt:off
    atoms = np.array([
            [0.0, 0.0,  0.0, 1],
            [1.0, 1.0,  0.0, 2],
            [1.0, 0.0,  0.0, 3],
            [2.0, 0.0, -1.0, 4]
    ])
    # fmt:on
    bonds = [(0, 2), (1, 2), (2, 3)]
    molecule = MoleculeGraph(atoms, bonds)

    new_atom = Atom([2.0, 2.0, 0.0], element=2)
    new_bonds = [0, 1, 1, 0]

    new_molecule = molecule.add_atom(new_atom, new_bonds)

    # fmt:off
    atoms_expected = np.array([
        [0.0, 0.0,  0.0, 1],
        [1.0, 1.0,  0.0, 2],
        [1.0, 0.0,  0.0, 3],
        [2.0, 0.0, -1.0, 4],
        [2.0, 2.0,  0.0, 2]
    ])
    # fmt:on
    bonds_expected = [(0, 2), (1, 2), (2, 3), (1, 4), (2, 4)]

    assert np.allclose(new_molecule.array(xyz=True, element=True), atoms_expected)
    assert len(new_molecule.bonds) == len(bonds_expected)
    for a, b in zip(new_molecule.bonds, bonds_expected):
        assert a == b

    assert np.allclose(molecule.array(xyz=True, element=True), atoms)
    assert len(molecule.bonds) == len(bonds)
    for a, b in zip(molecule.bonds, bonds):
        assert a == b


def test_GraphSeqStats():
    from mlspm.graph import GraphStats
    from mlspm.graph import MoleculeGraph

    classes = [[0], [1], [2]]

    # fmt:off
    pred = [
        MoleculeGraph(
            atoms = np.array([
                [0.0, 0.0, 0.0, 0],
                [0.0, 1.0, 0.0, 1],
                [1.0, 1.0, 0.0, 0]
            ]),
            bonds = [(0,1), (1,2)],
            classes=classes
        ),
        MoleculeGraph(
            atoms = np.array([
                [0.0, 1.0, 0.0, 0],
                [1.0, 1.0, 0.0, 1],
                [1.0, 1.0, 0.5, 2],
                [1.0, 2.0, 0.0, 2]
            ]),
            bonds = [(0,1), (0,2), (2,3)],
            classes=classes
        ),
        MoleculeGraph(
            atoms = np.array([
                [0.0, 1.0, 0.0, 0]
            ]),
            bonds = [],
            classes=classes
        ),
        MoleculeGraph(
            atoms = np.array([
                [0.0, 0.0, 0.0, 0]
            ]),
            bonds = [],
            classes=classes
        )
    ]

    ref = [
        MoleculeGraph(
            atoms = np.array([
                [0.0, 1.0, 0.3, 0],
                [0.0, 0.0, 0.0, 0]
            ]),
            bonds = [(0,1)],
            classes=classes
        ),
        MoleculeGraph(
            atoms = np.array([
                [0.0, 0.8, 0.0, 2],
                [2.0, 1.0, 0.0, 1],
                [1.1, 2.0, 0.0, 2],
                [1.0, 1.0, 0.3, 1]
            ]),
            bonds = [(0,1), (1,2), (0,2), (2,3)],
            classes=classes
        ),
        MoleculeGraph(
            atoms = np.array([
                [0.0, 0.0, 0.0, 0]
            ]),
            bonds = [],
            classes=classes
        ),
        MoleculeGraph(
            atoms = np.array([
                [0.0, 0.0, 0.0, 0]
            ]),
            bonds = [],
            classes=classes
        )
    ]

    seq_stats = GraphStats(classes=classes)
    seq_stats.add_batch(pred, ref)

    assert np.allclose(seq_stats.node_count_diffs(), [1, 0, 0, 0]), seq_stats.node_count_diffs()
    assert np.allclose(seq_stats.bond_count_diffs(), [1, -1, 0, 0]), seq_stats.bond_count_diffs()
    assert np.allclose(seq_stats.hausdorff_distances(), [np.sqrt(1+0.3**2), 1.0, 1.0, 0.0]), seq_stats.hausdorff_distances()
    assert np.allclose(seq_stats.matching_distances(), [0.3, 0.0, 0.2, 0.1, 0.2, 0.0]), seq_stats.matching_distances()
    assert np.allclose(seq_stats.extra_nodes().sum(), 3), seq_stats.extra_nodes()
    assert np.allclose(seq_stats.missing_nodes().sum(), 2), seq_stats.missing_nodes()
    assert np.allclose(seq_stats.total_nodes, 8), seq_stats.total_nodes
    assert np.allclose(seq_stats.total_samples, 4), seq_stats.total_samples
    assert np.allclose(seq_stats.largest_graph, 4), seq_stats.largest_graph
    assert np.allclose(seq_stats.conf_mat_node(), np.array([
        [2, 1, 0],
        [0, 0, 1],
        [1, 0, 1]
    ])), seq_stats.conf_mat_node()
    assert np.allclose(seq_stats.conf_mat_edge(), np.array([
        [0, 1],
        [1, 2],
    ])), seq_stats.conf_mat_edge()

    seq_stats = GraphStats(classes=classes, bin_size=1)
    seq_stats.add_batch(pred, ref)

    outdir = './test_stats'
    seq_stats.plot(outdir)
    seq_stats.report(outdir)
    shutil.rmtree(outdir)

    # fmt:on
