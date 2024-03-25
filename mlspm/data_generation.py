import io
import multiprocessing as mp
import os
import tarfile
import time
from multiprocessing.shared_memory import SharedMemory
from os import PathLike
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
from PIL import Image
from ppafm.ocl.field import ElectronDensity, HartreePotential


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


class TarSampleList(TypedDict, total=False):
    """
    - ``'hartree'``: Paths to the Hartree potentials. First item in the tuple is the path to the tar file,
      and second entry is a list of tar file member names.
    - ``'rho'``: (Optional) Paths to the electron densities. First item in the tuple is the path to the tar
      file, and second entry is a list tar file member names.
    - ``'rots'``: List of rotations for each sample.
    """

    hartree: tuple[PathLike, list[str]]
    rho: tuple[PathLike, list[str]]
    rots: list[np.ndarray]


class TarDataGenerator:
    """
    Iterable that loads data from tar archives with data saved in npz format for generating samples
    with the GeneratorAFMTrainer in ppafm.

    The npz files should contain the following entries:

        - ``'data'``: An array containing the potential/density on a 3D grid. The potential is assumed to be in
          units of eV and density in units of e/Å^3.
        - ``'origin'``: Lattice origin in 3D space as an array of shape ``(3,)``.
        - ``'lattice'``: Lattice vectors as an array of shape ``(3, 3)``, where the rows are the vectors.
        - ``'xyz'``: Atom xyz coordinates as an array of shape ``(n_atoms, 3)``.
        - ``'Z'``: Atom atomic numbers as an array of shape ``(n_atoms,)``.

    Yields dicts that contain the following:

        - ``'xyzs'``: Atom xyz coordinates.
        - ``'Zs'``: Atomic numbers.
        - ``'qs'``: Sample Hartree potential.
        - ``'rho_sample'``: Sample electron density if the sample dict contained ``rho``, or ``None`` otherwise.
        - ``'rot'``: Rotation matrix.

    Note: it is recommended to use ``multiprocessing.set_start_method('spawn')`` when using the :class:`TarDataGenerator`.
    Otherwise a lot of warnings about leaked memory objects may be thrown on exit.

    Arguments:
        samples: List of sample dicts as :class:`TarSampleList`. File paths should be relative to ``base_path``.
        base_path: Path to the directory with the tar files.
        n_proc: Number of parallel processes for loading data. The sample lists get divided evenly over the processes.
            For memory usage, note that a maximum number of samples double the number of processes can be loaded into
            memory at the same time.
        scale_pot: The loaded Hartree potentials are scaled by this factor in order to correct the units. The yielded potential should
            be in units of V. The default value of -1 works for potentials in units of eV.
        scale_rho: The loaded electron densities are scaled by this factor in order to correct the units. The yielded density should
            be in units of e/Å^3 with positive sign for the electron density.
    """

    _timings = False

    def __init__(
        self, samples: list[TarSampleList], base_path: PathLike = "./", n_proc: int = 1, scale_pot: float = -1, scale_rho: float = 1
    ):
        self.samples = samples
        self.base_path = Path(base_path)
        self.n_proc = n_proc
        self.scale_pot = scale_pot
        self.scale_rho = scale_rho
        self.pot = None
        self.rho = None

    def __len__(self) -> int:
        """Total number of samples (including rotations)"""
        return sum([sum([len(rots) for rots in sample_list["rots"]]) for sample_list in self.samples])

    def _launch_procs(self):
        queue_size = 2 * self.n_proc
        self.q = mp.Queue(queue_size)
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

    def _get_data(self, tar: tarfile.TarFile, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.load(tar.extractfile(name))
        array = data["data"]
        origin = data["origin"]
        lattice = data["lattice"]
        xyzs = data["xyz"]
        Zs = data["Z"]
        lvec = np.concatenate([origin[None, :], lattice], axis=0)
        return array, lvec, xyzs, Zs

    def _load_samples(self, sample_lists: list[TarSampleList], i_proc: int, event: mp.Event):

        proc_id = str(time.time_ns() + 1000000000 * i_proc)[-10:]
        print(f"Starting worker {i_proc}, id {proc_id}")

        start_time = time.perf_counter()
        total_bytes = 0
        n_sample_total = 0

        for sample_list in sample_lists:

            tar_path_hartree, name_list_hartree = sample_list["hartree"]
            tar_hartree = tarfile.open(self.base_path / tar_path_hartree, "r")

            n_sample = len(name_list_hartree)
            if len(sample_list["rots"]) != n_sample:
                raise ValueError(f"Inconsistent number of rotations in sample list ({len(sample_list['rots'])} != {n_sample})")

            use_rho = ("rho" in sample_list) and (sample_list["rho"] is not None)
            if use_rho:
                tar_path_rho, name_list_rho = sample_list["rho"]
                tar_rho = tarfile.open(self.base_path / tar_path_rho, "r")
                if len(name_list_rho) != n_sample:
                    raise ValueError(
                        f"Inconsistent number of samples between hartree and rho lists ({len(name_list_rho)} != {n_sample})"
                    )

            shm_pot_prev = None
            shm_rho_prev = None

            for i_sample in range(n_sample):

                if self._timings:
                    t0 = time.perf_counter()

                # Load data from tar(s)
                rots = sample_list["rots"][i_sample]
                pot, lvec_pot, xyzs, Zs = self._get_data(tar_hartree, name_list_hartree[i_sample])
                if not np.allclose(self.scale_pot, 1):
                    pot *= self.scale_pot
                total_bytes += pot.nbytes
                if use_rho:
                    rho, lvec_rho, _, _ = self._get_data(tar_rho, name_list_rho[i_sample])
                    if not np.allclose(self.scale_rho, 1):
                        rho *= self.scale_rho
                    rho_shape = rho.shape
                    total_bytes += rho.nbytes
                else:
                    lvec_rho = None
                    rho_shape = None

                if self._timings:
                    t1 = time.perf_counter()

                # Put the data to shared memory
                sample_id_pot = f"{i_proc}_{proc_id}_{i_sample}_pot"
                shm_pot = _put_to_shared_memory(pot, sample_id_pot)
                if use_rho:
                    sample_id_rho = f"{i_proc}_{proc_id}_{i_sample}_rho"
                    shm_rho = _put_to_shared_memory(rho, sample_id_rho)
                else:
                    sample_id_rho = None
                    shm_rho = None

                # Inform the main process of the data using the queue
                self.q.put((i_proc, sample_id_pot, sample_id_rho, pot.shape, rho_shape, lvec_pot, lvec_rho, xyzs, Zs, rots))

                if self._timings:
                    t2 = time.perf_counter()

                if i_sample > 0:
                    # Wait until main process is done with the previous data
                    _wait_and_unlink(i_proc, event, shm_pot_prev, shm_rho_prev)

                if self._timings:
                    t3 = time.perf_counter()
                    n_sample_total += 1
                    print(
                        f"[Worker {i_proc}, id {sample_id_pot}] Get data / Shm / Wait-unlink: "
                        f"{t1 - t0:.5f} / {t2 - t1:.5f} / {t3 - t2:.5f} "
                    )

                shm_pot_prev = shm_pot
                shm_rho_prev = shm_rho

            # Wait to unlink the last data
            _wait_and_unlink(i_proc, event, shm_pot, shm_rho)

            tar_hartree.close()
            if use_rho:
                tar_rho.close()

        if self._timings:
            dt = time.perf_counter() - start_time
            print(
                f"[Worker {i_proc}]: Loaded {n_sample_total} samples in {dt}s, totaling {total_bytes / 2**30:.3f}GiB. "
                f"Average load time: {dt / n_sample_total}s."
            )

    def _get_queue_sample(
        self,
    ) -> tuple[int, np.ndarray, np.ndarray, list[np.ndarray], HartreePotential, SharedMemory, ElectronDensity, SharedMemory, str]:

        if self._timings:
            t0 = time.perf_counter()

        i_proc, sample_id_pot, sample_id_rho, pot_shape, rho_shape, lvec_pot, lvec_rho, xyzs, Zs, rots = self.q.get(timeout=200)

        if self._timings:
            t1 = time.perf_counter()

        shm_pot = SharedMemory(sample_id_pot)
        pot = np.ndarray(pot_shape, dtype=np.float32, buffer=shm_pot.buf)
        if self.pot is None:
            self.pot = HartreePotential(pot, lvec_pot)
        else:
            self.pot.update_array(pot, lvec_pot)

        if self._timings:
            t2 = time.perf_counter()

        if sample_id_rho is not None:
            shm_rho = SharedMemory(sample_id_rho)
            rho = np.ndarray(rho_shape, dtype=np.float32, buffer=shm_rho.buf)
            if self.rho is None:
                self.rho = ElectronDensity(rho, lvec_rho)
            else:
                self.rho.update_array(rho, lvec_rho)
        else:
            shm_rho = None
            rho = None

        if self._timings:
            t3 = time.perf_counter()
            print(f"[Main, receive data, id {sample_id_pot}] Queue / Pot / Rho: " f"{t1 - t0:.5f} / {t2 - t1:.5f} / {t3 - t2:.5f}")

        return i_proc, xyzs, Zs, rots, self.pot, shm_pot, self.rho, shm_rho, sample_id_pot

    def _yield_samples(self):

        start_time = time.perf_counter()
        n_sample_yielded = 0

        n_sample_total = sum([len(sample_list["rots"]) for sample_list in self.samples])

        for _ in range(n_sample_total):

            if self._timings:
                t0 = time.perf_counter()

            i_proc, xyzs, Zs, rots, pot, shm_pot, rho, shm_rho, sample_id = self._get_queue_sample()
            if self._timings:
                t1 = time.perf_counter()

            for rot in rots:
                sample_dict = {"xyzs": xyzs, "Zs": Zs, "qs": pot, "rho_sample": rho, "rot": rot}
                yield sample_dict
                n_sample_yielded += 1

            if self._timings:
                t2 = time.perf_counter()

            # Close shared memory and inform producer that the shared memory can be unlinked
            shm_pot.close()
            if shm_rho is not None:
                shm_rho.close()
            self.events[i_proc].set()

            if self._timings:
                t3 = time.perf_counter()
                print(f"[Main, id {sample_id}] Receive data / Yield / Event: " f"{t1 - t0:.5f} / {t2 - t1:.5f} / {t3 - t2:.5f}")

        if self._timings:
            dt = time.perf_counter() - start_time
            print(f"[Main]: Yielded {n_sample_yielded} samples in {dt}s. Average yield time: {dt / n_sample_yielded}s.")


def _put_to_shared_memory(array, name):
    shm = SharedMemory(create=True, size=array.nbytes, name=name)
    b = np.ndarray(array.shape, dtype=np.float32, buffer=shm.buf)
    b[:] = array[:]
    return shm


def _wait_and_unlink(i_proc, event, shm_pot, shm_rho):
    if not event.wait(timeout=60):
        raise RuntimeError(f"[Worker {i_proc}]: Did not receive signal from main process in 60 seconds.")
    event.clear()
    shm_pot.close()
    shm_pot.unlink()
    if shm_rho:
        shm_rho.close()
        shm_rho.unlink()
