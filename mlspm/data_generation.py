import io
import multiprocessing as mp
import multiprocessing.shared_memory
import os
import tarfile
import time
from os import PathLike
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
from PIL import Image

class TarWriter:
    """
    Write samples of AFM images, molecules and descriptors to tar files. Use as a context manager and add samples with
    :meth:`add_sample`.

    Each tar file has a maximum number of samples, and whenever that maximum is reached, a new tar file is created.
    The generated tar files are named as ``{base_name}_{n}.tar`` and saved into the specified folder. The current tar file
    handle is always available in the attribute :attr:`ft`, and is automatically closed when the context ends.

    Arguments:
        base_path: Path to directory where tar files are saved.
        base_name: Base name for output tar files. The number of the tar file is appended to the name.
        max_count: Maximum number of samples per tar file.
        png_compress_level: Compression level 1-9 for saved png images. Larger value for smaller file size but slower
            write speed.
    """

    def __init__(self, base_path: PathLike = "./", base_name: str = "", max_count: int = 100, png_compress_level=4):
        self.base_path = Path(base_path)
        self.base_name = base_name
        self.max_count = max_count
        self.png_compress_level = png_compress_level

    def __enter__(self):
        self.sample_count = 0
        self.total_count = 0
        self.tar_count = 0
        self.ft = self._get_tar_file()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.ft.close()

    def _get_tar_file(self):
        file_path = self.base_path / f"{self.base_name}_{self.tar_count}.tar"
        if os.path.exists(file_path):
            raise RuntimeError(f"Tar file already exists at `{file_path}`")
        return tarfile.open(file_path, "w", format=tarfile.GNU_FORMAT)

    def add_sample(self, X: list[np.ndarray], xyzs: np.ndarray, Y: Optional[np.ndarray] = None, comment_str: str = ""):
        """
        Add a sample to the current tar file.

        Arguments:
            X: AFM images. Each list item corresponds to an AFM tip and is an array of shape (nx, ny, nz).
            xyzs: Atom coordinates and elements. Each row is one atom and is of the form [x, y, z, element].
            Y: Image descriptors. Each list item is one descriptor and is an array of shape (nx, ny).
            comment_str: Comment line (second line) to add to the xyz file.
        """

        if self.sample_count >= self.max_count:
            self.tar_count += 1
            self.sample_count = 0
            self.ft.close()
            self.ft = self._get_tar_file()

        # Write AFM images
        for i, x in enumerate(X):
            for j in range(x.shape[-1]):
                xj = x[:, :, j]
                xj = ((xj - xj.min()) / np.ptp(xj) * (2**8 - 1)).astype(np.uint8)  # Convert range to 0-255 integers
                img_bytes = io.BytesIO()
                Image.fromarray(xj.T[::-1], mode="L").save(img_bytes, "png", compress_level=self.png_compress_level)
                img_bytes.seek(0)  # Return stream to start so that addfile can read it correctly
                self.ft.addfile(get_tarinfo(f"{self.total_count}.{j:02d}.{i}.png", img_bytes), img_bytes)
                img_bytes.close()

        # Write xyz file
        xyz_bytes = io.BytesIO()
        xyz_bytes.write(bytearray(f"{len(xyzs)}\n{comment_str}\n", "utf-8"))
        for xyz in xyzs:
            xyz_bytes.write(bytearray(f"{int(xyz[-1])}\t", "utf-8"))
            for i in range(len(xyz) - 1):
                xyz_bytes.write(bytearray(f"{xyz[i]:10.8f}\t", "utf-8"))
            xyz_bytes.write(bytearray("\n", "utf-8"))
        xyz_bytes.seek(0)  # Return stream to start so that addfile can read it correctly
        self.ft.addfile(get_tarinfo(f"{self.total_count}.xyz", xyz_bytes), xyz_bytes)
        xyz_bytes.close()

        # Write image descriptors (if any)
        if Y is not None:
            for i, y in enumerate(Y):
                img_bytes = io.BytesIO()
                np.save(img_bytes, y.astype(np.float32))
                img_bytes.seek(0)  # Return stream to start so that addfile can read it correctly
                self.ft.addfile(get_tarinfo(f"{self.total_count}.desc_{i}.npy", img_bytes), img_bytes)
                img_bytes.close()

        self.sample_count += 1
        self.total_count += 1


def get_tarinfo(fname: str, file_bytes: io.BytesIO):
    info = tarfile.TarInfo(fname)
    info.size = file_bytes.getbuffer().nbytes
    info.mtime = time.time()
    return info

class TarSample(TypedDict, total=False):
    """
    - ``'hartree'``: Path to the Hartree potential. First item in the tuple is the path
      to the tar file relative to ``base_path``, and second entry is the tar file member name.
    - ``'rho'``: (Optional) Path to the electron density. First item in the tuple is the path
      to the tar file relative to ``base_path``, and second entry is the tar file member name.
    - ``'rots'``: List of rotations to generate for the sample.
    """
    hartree: tuple[str, str]
    rho: tuple[str, str]
    rots: list[np.ndarray]

class TarDataGenerator:
    """
    Iterable that loads data from tar archives with data saved in npz format for generating samples
    with the GeneratorAFMTrainer in ppafm.

    The npz files should contain the following entries:

        - ``'array'``: An array containing the potential/density on a 3D grid.
        - ``'origin'``: Lattice origin in 3D space as an array of shape ``(3,)``.
        - ``'lattice'``: Lattice vectors as an array of shape ``(3, 3)``, where the rows are the vectors.
        - ``'xyz'``: Atom xyz coordinates as an array of shape ``(n_atoms, 3)``.
        - ``'Z'``: Atom atomic numbers as an array of shape ``(n_atoms,)``.

    Arguments:
        samples: List of sample dicts as :class:`TarSample`. If ``rho`` is present in the dict, the full-density-based model
            is used in the simulation. Otherwise Lennard-Jones with Hartree electrostatics is used.
        base_path: Path to the directory with the tar files.
        n_proc: Number of parallel processes for loading data. The samples get divided evenly over the processes.
    """

    _timings = False

    def __init__(self, samples: list[TarSample], base_path: PathLike = "./", n_proc: int = 1):
        self.samples = samples
        self.base_path = Path(base_path)
        self.n_proc = n_proc

    def __len__(self):
        """Total number of samples (including rotations)"""
        return sum([len(s["rots"]) for s in self.samples])

    def _launch_procs(self):
        self.q = mp.Queue(maxsize=self.n_proc)
        self.events = []
        samples_split = np.array_split(self.samples, self.n_proc)
        for i in range(self.n_proc):
            event = mp.Event()
            p = mp.Process(target=self._load_samples, args=(samples_split[i], i, event))
            p.start()
            self.events.append(event)

    def __iter__(self):
        self._launch_procs()
        self.iterator = iter(self._yield_samples())
        return self

    def __next__(self):
        return next(self.iterator)

    def _get_data(self, tar_path: PathLike, name: str):
        tar_path = self.base_path / tar_path
        with tarfile.open(tar_path, "r") as f:
            data = np.load(f.extractfile(name))
            array = data["data"]
            origin = data["origin"]
            lattice = data["lattice"]
            xyzs = data["xyz"]
            Zs = data["Z"]
            lvec = np.concatenate([origin[None, :], lattice], axis=0)
        return array, lvec, xyzs, Zs

    def _load_samples(self, samples: list[TarSample], i_proc: int, event: mp.Event):

        proc_id = str(time.time_ns() + 1000000000 * i_proc)[-10:]
        print(f"Starting worker {i_proc}, id {proc_id}")

        for i, sample in enumerate(samples):

            if self._timings:
                t0 = time.perf_counter()

            # Load data from tar(s)
            rots = sample["rots"]
            hartree_tar_path, name = sample["hartree"]
            pot, lvec, xyzs, Zs = self._get_data(hartree_tar_path, name)
            pot *= -1  # Unit conversion, eV -> V
            if "rho" in sample:
                rho_tar_path, name = sample["rho"]
                rho, _, _, _ = self._get_data(rho_tar_path, name)

            if self._timings:
                t1 = time.perf_counter()

            # Put the data to shared memory
            sample_id_pot = f"{i_proc}_{proc_id}_{i}_pot"
            shm_pot = mp.shared_memory.SharedMemory(create=True, size=pot.nbytes, name=sample_id_pot)
            b = np.ndarray(pot.shape, dtype=np.float32, buffer=shm_pot.buf)
            b[:] = pot[:]

            if "rho" in sample:
                sample_id_rho = f"{i_proc}_{proc_id}_{i}__rho"
                shm_rho = mp.shared_memory.SharedMemory(create=True, size=rho.nbytes, name=sample_id_rho)
                b = np.ndarray(rho.shape, dtype=np.float32, buffer=shm_rho.buf)
                b[:] = rho[:]
                rho_shape = rho.shape
            else:
                sample_id_rho = None
                rho_shape = None

            if self._timings:
                t2 = time.perf_counter()

            # Inform the main process of the data using the queue
            self.q.put((i_proc, sample_id_pot, sample_id_rho, pot.shape, rho_shape, lvec, xyzs, Zs, rots))

            if self._timings:
                t3 = time.perf_counter()

            # Wait until main process is done with the data
            if not event.wait(timeout=60):
                raise RuntimeError(f"[Worker {i_proc}]: Did not receive signal from main process in 60 seconds.")
            event.clear()

            if self._timings:
                t4 = time.perf_counter()

            # Done with shared memory
            shm_pot.unlink()
            shm_pot.close()
            if "rho" in sample:
                shm_rho.unlink()
                shm_rho.close()

            if self._timings:
                t5 = time.perf_counter()
                print(
                    f"[Worker {i_proc}, id {sample_id_pot}] Get data / Shm / Queue / Wait / Unlink: "
                    f"{t1 - t0:.5f} / {t2 - t1:.5f} / {t3 - t2:.5f} / {t4 - t3:.5f} / {t5 - t4:.5f}"
                )

    def _yield_samples(self):

        from ppafm.ocl.field import ElectronDensity, HartreePotential

        for _ in range(len(self)):

            if self._timings:
                t0 = time.perf_counter()

            # Get data info from the queue
            i_proc, sample_id_pot, sample_id_rho, pot_shape, rho_shape, lvec, xyzs, Zs, rots = self.q.get(timeout=200)

            # Get data from the shared memory
            shm_pot = mp.shared_memory.SharedMemory(sample_id_pot)
            pot = np.ndarray(pot_shape, dtype=np.float32, buffer=shm_pot.buf)
            pot = HartreePotential(pot, lvec)
            if sample_id_rho is not None:
                shm_rho = mp.shared_memory.SharedMemory(sample_id_rho)
                rho = np.ndarray(rho_shape, dtype=np.float32, buffer=shm_rho.buf)
                rho = ElectronDensity(rho, lvec)
            else:
                rho = None

            if self._timings:
                t1 = time.perf_counter()

            for rot in rots:
                sample_dict = {"xyzs": xyzs, "Zs": Zs, "qs": pot, "rho_sample": rho, "rot": rot}
                yield sample_dict

            if self._timings:
                t2 = time.perf_counter()

            # Close shared memory and inform producer that the shared memory can be unlinked
            shm_pot.close()
            if sample_id_rho is not None:
                shm_rho.close()
            self.events[i_proc].set()

            if self._timings:
                t3 = time.perf_counter()
                print(f"[Main, id {sample_id_pot}] Receive data / Yield / Event: " f"{t1 - t0:.5f} / {t2 - t1:.5f} / {t3 - t2:.5f}")
