import io
from pathlib import Path
from shutil import rmtree
import tarfile
import numpy as np
import pytest


def test_tar_writer():

    from mlspm.data_generation import TarWriter

    base_path = Path("./test_writer")
    base_name = "test"

    base_path.mkdir(exist_ok=True)

    with TarWriter(base_path, base_name, max_count=10) as tar_writer:
        for _ in range(20):
            X = [np.random.rand(128, 128, 10), np.random.rand(128, 128, 10)]
            Y = [np.random.rand(128, 128), np.random.rand(128, 128)]
            xyzs = np.concatenate([np.random.rand(10, 3), np.random.randint(1, 10, (10, 1))], axis=1)
            tar_writer.add_sample(X, xyzs, Y, comment_str="test comment")

    assert (base_path / "test_0.tar").exists()
    assert (base_path / "test_1.tar").exists()

    with tarfile.open(base_path / "test_0.tar") as ft:
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


def test_tar_data_generator():

    from mlspm.data_generation import TarDataGenerator, get_tarinfo
    from ppafm.ocl.oclUtils import init_env
    import multiprocessing as mp

    # Without this will throw a lot of warnings on exit
    # mp.set_start_method('spawn')

    # Loading data into HartreePotential etc. in TarDataGenerator requires for the pyopencl context to be setup
    init_env(i_platform=0)

    # Make dummy data
    tar_path_hartree = Path("./test_hartree.tar")
    tar_path_rho = Path("./test_rho.tar")
    n_sample = 5
    with tarfile.open(tar_path_hartree, "w") as f_hartree, tarfile.open(tar_path_rho, "w") as f_rho:

        hartrees = []
        rhos = []
        xyzs = []
        Zs = []
        lvecs = []
        rots = []
        names = []

        for i_sample in range(n_sample):

            hartree = np.random.rand(10, 15, 12).astype(np.float32)
            rho = np.random.rand(12, 10, 8)
            xyz = np.random.rand(10, 3)
            Z = np.random.rand(10)
            lvec = np.random.rand(4, 3)
            rot = np.random.rand(3, 3)

            hartree_bytes = io.BytesIO()
            rho_bytes = io.BytesIO()
            np.savez(hartree_bytes, data=hartree, origin=lvec[0], lattice=lvec[1:], xyz=xyz, Z=Z)
            np.savez(rho_bytes, data=rho, origin=lvec[0], lattice=lvec[1:], xyz=xyz, Z=Z)
            hartree_bytes.seek(0)
            rho_bytes.seek(0)

            name = f"{i_sample}.npz"
            f_hartree.addfile(get_tarinfo(name, hartree_bytes), hartree_bytes)
            f_rho.addfile(get_tarinfo(name, rho_bytes), rho_bytes)

            hartrees.append(hartree)
            rhos.append(rho)
            xyzs.append(xyz)
            Zs.append(Z)
            lvecs.append(lvec)
            rots.append([rot])
            names.append(name)

    sample_list_fdbm = [
        {
            "hartree": (tar_path_hartree, names),
            "rho": (tar_path_rho, names),
            "rots": rots,
        }
    ]

    generator = TarDataGenerator(sample_list_fdbm, base_path='./', n_proc=1)

    for i_sample, sample in enumerate(generator):
        assert np.allclose(sample['xyzs'], xyzs[i_sample])
        assert np.allclose(sample['Zs'], Zs[i_sample])
        assert np.allclose(sample['rot'], rots[i_sample])
        assert np.allclose(sample['qs'].array, -hartrees[i_sample])
        assert np.allclose(sample['qs'].lvec, lvecs[i_sample])
        assert np.allclose(sample['rho_sample'].array, rhos[i_sample])
        assert np.allclose(sample['rho_sample'].lvec, lvecs[i_sample])


    sample_list_hartree = [
        {
            "hartree": (tar_path_hartree, names),
            "rho": None,
            "rots": rots,
        }
    ]

    generator = TarDataGenerator(sample_list_hartree, base_path='./', n_proc=1)

    for i_sample, sample in enumerate(generator):
        assert np.allclose(sample['xyzs'], xyzs[i_sample])
        assert np.allclose(sample['Zs'], Zs[i_sample])
        assert np.allclose(sample['rot'], rots[i_sample])
        assert np.allclose(sample['qs'].array, -hartrees[i_sample])
        assert np.allclose(sample['qs'].lvec, lvecs[i_sample])

    tar_path_hartree.unlink()
    tar_path_rho.unlink()
