import os
import tarfile
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

# fmt: off
DATASET_URLS = {
    "AFM-ice-Cu111": "https://zenodo.org/records/10047850/files/AFM-ice-Cu111.tar.gz?download=1"
}
# fmt: on


def _print_progress(block_num, block_size, total_size):
    if total_size == -1:
        return
    delta = block_size / total_size * 100
    current_size = block_num * block_size
    percent = current_size / total_size * 100
    percent_int = int(percent)
    if (percent - percent_int) > 1.0001 * delta:
        # Only print when crossing an integer percentage
        return
    if block_num > 0:
        print("\b\b\b", end="", flush=True)
    if current_size < total_size:
        print(f"{percent_int:2d}%", end="", flush=True)
    else:
        print("Done")


def _is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def _safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members, numeric_owner=numeric_owner)


def download_dataset(name: str, target_dir: PathLike):
    """
    Download and unpack a dataset to a target directory.

    The following datasets are available:
        - 'AFM-ice-Cu111': https://doi.org/10.5281/zenodo.10047850

    Arguments:
        name: Name of dataset to download.
        target_dir: Directory where dataset will be unpacked into. The directory and its parents will be created
            if they do not exist already. If the directory already exists and is not empty, then the operation
            is aborted.
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
            base_dir = os.path.commonprefix(ft.getnames())
            for m in ft.getmembers():
                if m.isfile():
                    # relative_to(base_dir) here gets rid of a common parent directory within the archive (if any),
                    # which makes it so that we can just directly extract the files to the target directory.
                    m.name = Path(m.name).relative_to(base_dir)
                    members.append(m)
            print(f"Extracting dataset to `{target_dir}`: ", end="", flush=True)
            for i, m in enumerate(members):
                _print_progress(i, 1, len(members) - 1)
                _safe_extract(ft, target_dir, [m])
