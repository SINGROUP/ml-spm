import numpy as np


def test_add_noise():
    from mlspm.preprocessing import add_noise

    np.random.seed(0)
    n = 1000
    c = 0.1

    for _ in range(n):
        test_input = np.zeros((3, 10))
        # The add_noise function scales the noise amplitude by the range of the data so we need at least one element
        # different from zero in all batch elements
        test_input[:, 0] = 1
        add_noise([test_input], c=c, randomize_amplitude=False)
        assert (test_input < (c + 1)).all()
        assert (np.abs(test_input) > 0).any()


def test_add_norm():
    from mlspm.preprocessing import add_norm

    # fmt: off
    test_input = [
        np.array([
            [0.5, -0.5],
            [0.5, -0.5]
        ]),
    ]
    # fmt: on

    add_norm(test_input, per_layer=False)
    test_output = test_input

    # fmt: off
    expected_output = [
        np.array([
            [1.0, -1.0],
            [1.0, -1.0]
        ])
    ]
    # fmt: on

    assert len(test_output) == len(expected_output)
    for a, b in zip(test_output, expected_output):
        assert np.allclose(a, b)

    # fmt: off
    test_input = [
        np.array([
            [[-0.5,  0.5],
             [-1.5, -0.5]],
            [[ 1.0,  1.0],
             [ 3.0,  5.0]]
        ]),
    ]
    # fmt: on

    add_norm(test_input, per_layer=True)
    test_output = test_input

    # fmt: off
    expected_output = [
        np.array([
            [[ 1.0,  1.0],
             [-1.0, -1.0]],
            [[-1.0, -1.0],
             [ 1.0,  1.0]],
        ]),
    ]
    # fmt: on

    assert len(test_output) == len(expected_output)
    for a, b in zip(test_output, expected_output):
        assert np.allclose(a, b)


def test_interpolate_and_crop():
    from mlspm.preprocessing import interpolate_and_crop

    X = interpolate_and_crop(Xs=[np.random.rand(2, 20, 22, 10)], real_dim=(10, 11), target_res=1, target_multiple=2)

    assert len(X) == 1
    assert X[0].shape == (2, 10, 10, 10)


def test_add_cutout():
    from mlspm.preprocessing import add_cutout

    np.random.seed(0)

    for _ in range(100):
        test_input = np.ones((3, 32, 32, 32))
        add_cutout([test_input], n_holes=10)
        assert np.isclose(test_input, 0.0).any()
