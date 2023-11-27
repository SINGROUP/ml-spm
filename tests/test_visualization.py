import shutil
from pathlib import Path

import numpy as np


def test_make_input_plots():

    from mlspm.visualization import make_input_plots

    save_dir = Path('test_input_plots')

    X = [np.random.rand(2, 20, 20, 2), np.random.rand(2, 20, 20, 2)]
    make_input_plots(X, outdir=save_dir)

    assert len(list(save_dir.glob('*.png'))) == 4

    shutil.rmtree(save_dir)

def test_plot_graphs():

    from mlspm.graph import MoleculeGraph
    from mlspm.visualization import plot_graphs

    save_dir = Path('test_graph_plots')

    # fmt:off
    atoms = [
        np.array([
            [0.1, 0.2, 0.3, 1],
            [0.5, 0.6, 0.7, 1],
            [0.9, 1.0, 1.1, 6],
            [1.3, 1.4, 1.5, 6]
        ]),
        np.array([
            [1.7, 1.8, 1.9, 1],
            [2.1, 2.2, 2.3, 6]
        ]),
        np.empty((0, 4)),
        np.array([
            [2.5, 2.6, 2.7, 1],
            [2.9, 3.0, 3.1, 6],
            [3.3, 3.4, 3.5, 6]
        ]),
    ]
    # fmt:on
    bonds = [
        [(0,1), (0,2), (1,3), (2,3)],
        [(0,1)],
        [],
        [(0,1), (1,2)]
    ]
    classes = [[1], [6]]
    mols = [MoleculeGraph(a, b, classes) for a, b in zip(atoms, bonds)]
    box_borders = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])

    plot_graphs(mols, mols, box_borders=box_borders, classes=classes, outdir=save_dir)

    assert len(list(save_dir.glob('*.png'))) == 4

    shutil.rmtree(save_dir)

def test_plot_distribution_grid():

    from mlspm.graph import make_position_distribution, MoleculeGraph
    from mlspm.visualization import plot_distribution_grid

    save_dir = Path('test_distribution_plots')

    # fmt:off
    atoms = np.array([
        [2.0, 2.0, 1.0, 0],
        [2.0, 6.0, 2.0, 0],
        [6.0, 4.0, 2.0, 0],
        [4.0, 2.0, 2.0, 0]
    ])
    # fmt:on
    box_borders = np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 3.0]])
    mols = [MoleculeGraph(atoms, bonds=[])]
    std = 0.2

    dist = make_position_distribution(mols, box_borders=box_borders, std=std)

    plot_distribution_grid(dist, dist, box_borders=box_borders, outdir=save_dir)

    assert len(list(save_dir.glob('*.png'))) == 2

    shutil.rmtree(save_dir)
