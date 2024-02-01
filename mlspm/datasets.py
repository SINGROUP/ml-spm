import os
import tarfile
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from .utils import _print_progress

DATASET_URLS = {
    "AFM-ice-Cu111": "https://zenodo.org/records/10047850/files/AFM-ice-Cu111.tar.gz?download=1",
    "AFM-ice-Au111-monolayer": "https://zenodo.org/records/10049832/files/AFM-ice-Au111-monolayer.tar.gz?download=1",
    "AFM-ice-Au111-bilayer": "https://zenodo.org/records/10049856/files/AFM-ice-Au111-bilayer.tar.gz?download=1",
    "AFM-ice-exp": "https://zenodo.org/records/10054847/files/exp_data_ice.tar.gz?download=1",
    "AFM-ice-relaxed": "https://zenodo.org/records/10362511/files/relaxed_structures.tar.gz?download=1",
    "ASD-AFM-molecules": "https://zenodo.org/records/10562769/files/molecules.tar.gz?download=1",
    "AFM-camphor-exp": "https://zenodo.org/records/10562769/files/afm_camphor.tar.gz?download=1",
    "ED-AFM-molecules": "https://zenodo.org/records/10606443/files/molecules_rebias.tar.gz?download=1",
}


def _common_parent(paths):
    path_parts = [list(Path(p).parts) for p in paths]
    common_part = Path()
    for parts in zip(*path_parts):
        p = parts[0]
        if all(part == p for part in parts):
            common_part /= p
        else:
            break
    return common_part


def download_dataset(name: str, target_dir: PathLike):
    """
    Download and unpack a dataset to a target directory.

    The following datasets are available:

        - ``'AFM-ice-Cu111'``: https://doi.org/10.5281/zenodo.10047850
        - ``'AFM-ice-Au111-monolayer'``: https://doi.org/10.5281/zenodo.10049832
        - ``'AFM-ice-Au111-bilayer'``: https://doi.org/10.5281/zenodo.10049856
        - ``'AFM-ice-exp'``: https://doi.org/10.5281/zenodo.10054847
        - ``'AFM-ice-relaxed'``: https://doi.org/10.5281/zenodo.10362511
        - ``'ASD-AFM-molecules'``: https://doi.org/10.5281/zenodo.10562769 - 'molecules.tar.gz'
        - ``'AFM-camphor-exp'``: https://doi.org/10.5281/zenodo.10562769 - 'afm_camphor.tar.gz'
        - ``'ED-AFM-molecules'``: https://doi.org/10.5281/zenodo.10606443

    Arguments:
        name: Name of the dataset to download.
        target_dir: Directory where the dataset will be unpacked into. The directory and its parents will be created if they
            do not exist already. If the directory already exists and is not empty, then the operation is aborted.
    """
    try:
        dataset_url = DATASET_URLS[name]
    except KeyError:
        raise ValueError(f"Unrecognized dataset name `{name}`")

    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Target directory `{target_dir}` exists and is not empty. Skipping downloading dataset `{name}`.")
        return

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / f"dataset_{name}.tar.gz"
        print(f"Downloading dataset `{name}`: ", end="")
        urlretrieve(dataset_url, temp_file, _print_progress)
        target_dir.mkdir(exist_ok=True, parents=True)
        with tarfile.open(temp_file, "r") as ft:
            print("Reading archive files...")
            members = []
            base_dir = _common_parent(ft.getnames())
            for m in ft.getmembers():
                if m.isfile():
                    # relative_to(base_dir) here gets rid of a common parent directory within the archive (if any),
                    # which makes it so that we can just directly extract the files to the target directory.
                    m.name = Path(m.name).relative_to(base_dir)
                    members.append(m)
            print(f"Extracting dataset to `{target_dir}`: ", end="", flush=True)
            for i, m in enumerate(members):
                _print_progress(i, 1, len(members) - 1)
                ft.extract(m, target_dir)
