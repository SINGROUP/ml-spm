from os import PathLike
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from .utils import _print_progress

WEIGHTS_URLS = {
    "graph-ice-cu111": "https://zenodo.org/records/10054348/files/weights_ice-cu111.pth?download=1",
    "graph-ice-au111-monolayer": "https://zenodo.org/records/10054348/files/weights_ice-au111-monolayer.pth?download=1",
    "graph-ice-au111-bilayer": "https://zenodo.org/records/10054348/files/weights_ice-au111-bilayer.pth?download=1",
    "asdafm-light": "https://zenodo.org/records/10514470/files/weights_asdafm_light.pth?download=1",
    "asdafm-heavy": "https://zenodo.org/records/10514470/files/weights_asdafm_heavy.pth?download=1",
    "edafm-base": "https://zenodo.org/records/10606273/files/base.pth?download=1",
    "edafm-single-channel": "https://zenodo.org/records/10606273/files/single-channel.pth?download=1",
    "edafm-CO-Cl": "https://zenodo.org/records/10606273/files/CO-Cl.pth?download=1",
    "edafm-Xe-Cl": "https://zenodo.org/records/10606273/files/Xe-Cl.pth?download=1",
    "edafm-constant-noise": "https://zenodo.org/records/10606273/files/constant-noise.pth?download=1",
    "edafm-uniform-noise": "https://zenodo.org/records/10606273/files/uniform_noise.pth?download=1",
    "edafm-no-gradient": "https://zenodo.org/records/10606273/files/no-gradient.pth?download=1",
    "edafm-matched-tips": "https://zenodo.org/records/10606273/files/matched-tips.pth?download=1",
}


def download_weights(weights_name: str, target_path: Optional[PathLike] = None) -> PathLike:
    """
    Download pretrained weights for models.

    The following weights are available:

        - ``'graph-ice-cu111'``: PosNet trained on ice clusters on Cu(111). (https://doi.org/10.5281/zenodo.10054348)
        - ``'graph-ice-au111-monolayer'``: PosNet trained on monolayer ice clusters on Au(111).
          (https://doi.org/10.5281/zenodo.10054348)
        - ``'graph-ice-au111-bilayer'``: PosNet trained on bilayer ice clusters on Au(111).
          (https://doi.org/10.5281/zenodo.10054348)
        - ``'asdafm-light'``: :class:`.ASDAFMNet` trained on molecules containing the elements H, C, N, O, and F.
          (https://doi.org/10.5281/zenodo.10514470)
        - ``'asdafm-heavy'``: :class:`.ASDAFMNet` trained on molecules additionally containing Si, P, S, Cl, and Br.
          (https://doi.org/10.5281/zenodo.10514470)
        - ``'edafm-base'``: :class:`.EDAFMNet` used for all predictions in the main ED-AFM paper and used for comparison in
          the various tests in the supplementary information of the paper. (https://doi.org/10.5281/zenodo.10606273)
        - ``'edafm-single-channel'``: :class:`.EDAFMNet` trained on only a single CO-tip AFM input.
          (https://doi.org/10.5281/zenodo.10606273)
        - ``'edafm-CO-Cl'``: :class:`.EDAFMNet` trained on alternative tip combination of CO and Cl.
          (https://doi.org/10.5281/zenodo.10606273)
        - ``'edafm-Xe-Cl'``: :class:`.EDAFMNet` trained on alternative tip combination of Xe and Cl.
          (https://doi.org/10.5281/zenodo.10606273)
        - ``'edafm-constant-noise'``: :class:`.EDAFMNet` trained using constant noise amplitude instead of normally distributed
          amplitude. (https://doi.org/10.5281/zenodo.10606273)
        - ``'edafm-uniform-noise'``: :class:`.EDAFMNet` trained using uniform random noise amplitude instead of normally
          distributed amplitude. (https://doi.org/10.5281/zenodo.10606273)
        - ``'edafm-no-gradient'``: :class:`.EDAFMNet` trained without background-gradient augmentation.
          (https://doi.org/10.5281/zenodo.10606273)
        - ``'edafm-matched-tips'``: :class:`.EDAFMNet` trained on data with matched tip distance between CO and Xe,
          instead of independently randomized distances. (https://doi.org/10.5281/zenodo.10606273)

    Arguments:
        weights_name: Name of weights to download.
        target_path: Path where the weights file will be saved. If specified, the parent directory for the file has to exists.
            If not specified, a location in a cache directory is chosen. If the target file already exists, the download is skipped

    Returns:
        Path where the weights were saved.
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
