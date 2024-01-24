
from pathlib import Path
from shutil import rmtree
import tarfile
import numpy as np
import pytest


def test_tar_writer():

    from mlspm.data_generation import TarWriter

    base_path = Path('./test_writer')
    base_name = 'test'

    base_path.mkdir(exist_ok=True)

    with TarWriter(base_path, base_name, max_count=10) as tar_writer:
        for _ in range(20):
            X = [np.random.rand(128, 128, 10), np.random.rand(128, 128, 10)]
            Y = [np.random.rand(128, 128), np.random.rand(128, 128)]
            xyzs = np.concatenate([np.random.rand(10, 3), np.random.randint(1, 10, (10, 1))], axis=1)
            tar_writer.add_sample(X, xyzs, Y, comment_str='test comment')

    assert (base_path / 'test_0.tar').exists()
    assert (base_path / 'test_1.tar').exists()

    with tarfile.open(base_path / 'test_0.tar') as ft:
        names = [m.name for m in ft.getmembers()]
        assert len(names) == 10 * (2 * 10 + 1 + 2)
        assert "0.00.0.png" in names
        assert "0.00.1.png" in names
        assert "0.09.1.png" in names
        assert "0.00.2.png" not in names
        assert "0.10.0.png" not in names
        assert "0.xyz" in names
        assert "0.desc_0.npy" in names

    with pytest.raises(RuntimeError):
        # Cannot overwrite an existing file
        with TarWriter(base_path, base_name, max_count=10) as tar_writer:
            pass

    rmtree(base_path)
        
test_tar_writer()