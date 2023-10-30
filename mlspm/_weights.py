from os import PathLike
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from .utils import _print_progress

WEIGHTS_URLS = {
    "graph-ice-cu111": "https://zenodo.org/records/10054348/files/weights_ice-cu111.pth?download=1",
    "graph-ice-au111-monolayer": "https://zenodo.org/records/10054348/files/weights_ice-au111-monolayer.pth?download=1",
    "graph-ice-au111-bilayer": "https://zenodo.org/records/10054348/files/weights_ice-au111-bilayer.pth?download=1",
}


def download_weights(weights_name: str, target_path: Optional[PathLike] = None) -> PathLike:
    """
    Download pretrained weights for models.

    The following weights are available:
        - 'graph-ice-cu111': PosNet trained on ice clusters on Cu(111).
        - 'graph-ice-au111-monolayer': PosNet trained on monolayer ice clusters on Au(111).
        - 'graph-ice-au111-bilayer': PosNet trained on bilayer ice clusters on Au(111).

    Arguments:
        weights_name: Name of weights to download.
        target_path: Path where the weights file will be saved. If specified, the parent directory for the file has to exists.
            If not specified, a location in cache directory is chosen. If the target file already exists, the download is skipped

    Returns: path where the weights were saved.

    """
    try:
        weights_url = WEIGHTS_URLS[weights_name]
    except KeyError:
        raise ValueError(f"Unrecognized weights name `{weights_name}`")

    if target_path is None:
        cache_dir = Path.home() / ".cache" / "mlspm"
        cache_dir.mkdir(exist_ok=True, parents=True)
        target_path = cache_dir / f"{weights_name}.pth"
    else:
        target_path = Path(target_path)

    if target_path.exists():
        print(f"Target path `{target_path}` already exists. Skipping downloading weights `{weights_name}`.")
        return target_path

    print(f"Downloading weights `{weights_name}`: ", end="")
    urlretrieve(weights_url, target_path, _print_progress)

    return target_path
