
import io
import os
import tarfile
import time
from os import PathLike
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


class TarWriter:
    '''
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
    '''

    def __init__(self, base_path: PathLike='./', base_name: str='', max_count: int=100, png_compress_level=4):
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
        file_path = self.base_path / f'{self.base_name}_{self.tar_count}.tar'
        if os.path.exists(file_path):
            raise RuntimeError(f'Tar file already exists at `{file_path}`')
        return tarfile.open(file_path, 'w', format=tarfile.GNU_FORMAT)

    def add_sample(self, X: List[np.ndarray], xyzs: np.ndarray, Y: Optional[np.ndarray]=None, comment_str: str=''):
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
                xj = ((xj - xj.min()) / np.ptp(xj) * (2**8 - 1)).astype(np.uint8) # Convert range to 0-255 integers
                img_bytes = io.BytesIO()
                Image.fromarray(xj.T[::-1], mode='L').save(img_bytes, 'png', compress_level=self.png_compress_level)
                img_bytes.seek(0) # Return stream to start so that addfile can read it correctly
                self.ft.addfile(get_tarinfo(f'{self.total_count}.{j:02d}.{i}.png', img_bytes), img_bytes)
                img_bytes.close()
        
        # Write xyz file
        xyz_bytes = io.BytesIO()
        xyz_bytes.write(bytearray(f'{len(xyzs)}\n{comment_str}\n', 'utf-8'))
        for xyz in xyzs:
            xyz_bytes.write(bytearray(f'{int(xyz[-1])}\t', 'utf-8'))
            for i in range(len(xyz)-1):
                xyz_bytes.write(bytearray(f'{xyz[i]:10.8f}\t', 'utf-8'))
            xyz_bytes.write(bytearray('\n', 'utf-8'))
        xyz_bytes.seek(0) # Return stream to start so that addfile can read it correctly
        self.ft.addfile(get_tarinfo(f'{self.total_count}.xyz', xyz_bytes), xyz_bytes)
        xyz_bytes.close()

        # Write image descriptors (if any)
        if Y is not None:
            for i, y in enumerate(Y):
                img_bytes = io.BytesIO()
                np.save(img_bytes, y.astype(np.float32))
                img_bytes.seek(0) # Return stream to start so that addfile can read it correctly
                self.ft.addfile(get_tarinfo(f'{self.total_count}.desc_{i}.npy', img_bytes), img_bytes)
                img_bytes.close()

        self.sample_count += 1
        self.total_count += 1

def get_tarinfo(fname: str, file_bytes: io.BytesIO):
    info = tarfile.TarInfo(fname)
    info.size = file_bytes.getbuffer().nbytes
    info.mtime = time.time()
    return info