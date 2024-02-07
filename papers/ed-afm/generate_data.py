import time
from math import ceil
from pathlib import Path

import numpy as np
from ppafm.common import eVA_Nm
from ppafm.ml.AuxMap import ESMapConstant
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl.AFMulator import AFMulator

from mlspm.data_generation import TarWriter
from mlspm.datasets import download_dataset


class Trainer(InverseAFMtrainer):

    def on_afm_start(self):
        # Use different lateral stiffness for Cl than CO and Xe
        if self.afmulator.iZPP in [8, 54]:
            afmulator.scanner.stiffness = np.array([0.25, 0.25, 0.0, 30.0], dtype=np.float32) / -eVA_Nm
        elif self.afmulator.iZPP == 17:
            afmulator.scanner.stiffness = np.array([0.5, 0.5, 0.0, 30.0], dtype=np.float32) / -eVA_Nm
        else:
            raise RuntimeError(f"Unknown tip {self.afmulator.iZPP}")

    # Override to randomize tip distance and probe tilt
    def handle_distance(self):
        self.randomize_distance(delta=0.25)
        self.randomize_tip(max_tilt=0.5)
        super().handle_distance()


if __name__ == "__main__":

    # Path where molecule geometry files are saved
    mol_dir = Path("./molecules")

    # Directory where to save data
    data_dir = Path(f"./data/")

    # Define simulator and image descriptor parameters
    scan_window = ((0, 0, 6.0), (23.875, 23.875, 7.9))
    scan_dim = (192, 192, 19)
    afmulator = AFMulator(pixPerAngstrome=10, scan_dim=scan_dim, scan_window=scan_window, tipR0=[0.0, 0.0, 4.0])
    aux_maps = [
        ESMapConstant(
            scan_dim=afmulator.scan_dim[:2],
            scan_window=[afmulator.scan_window[0][:2], afmulator.scan_window[1][:2]],
            height=4.0,
            vdW_cutoff=-2.0,
            Rpp=1.0,
        )
    ]
    generator_arguments = {
        "afmulator": afmulator,
        "aux_maps": aux_maps,
        "batch_size": 1,
        "distAbove": 5.25,
        "iZPPs": [8, 54, 17],  # CO, Xe, Cl
        "Qs": [[-10, 20, -10, 0], [30, -60, 30, 0], [-0.3, 0, 0, 0]],
        "QZs": [[0.1, 0, -0.1, 0], [0.1, 0, -0.1, 0], [0, 0, 0, 0]],
    }

    # Number of tar file shards for each set
    target_shard_count = 8

    # Make sure the save directory exists
    data_dir.mkdir(exist_ok=True, parents=True)

    # Download the dataset. The extraction may take a while since there are ~235k files.
    download_dataset("ED-AFM-molecules", mol_dir)

    # Paths to molecule xyz files
    train_paths = list((mol_dir / "train").glob("*.xyz"))
    val_paths = list((mol_dir / "validation").glob("*.xyz"))
    test_paths = list((mol_dir / "test").glob("*.xyz"))

    # Generate dataset
    start_time = time.perf_counter()
    counter = 1
    for mode, molecules in zip(["train", "val", "test"], [train_paths, val_paths, test_paths]):

        # Construct generator
        generator = Trainer(paths=molecules, **generator_arguments)

        # Generate data
        max_count = ceil(len(generator) / target_shard_count)
        start_gen = time.perf_counter()
        with TarWriter(data_dir, f"{data_dir.name}-K-0_{mode}", max_count=max_count) as tar_writer:
            for i, (X, Y, xyz) in enumerate(generator):

                # Get rid of the batch dimension
                X = [x[0] for x in X]
                Y = [y[0] for y in Y]
                xyz = xyz[0]

                # Save information of the simulation parameters into the xyz comment line
                amp = generator.afmulator.amplitude
                R0 = generator.afmulator.tipR0
                kxy = generator.afmulator.scanner.stiffness[0]
                sw = generator.afmulator.scan_window
                comment_str = f"Scan window: [{sw[0]}, {sw[1]}], Amplitude: {amp}, tip R0: {R0}, kxy: {kxy}"

                # Write the sample to a tar file
                tar_writer.add_sample(X, xyz, Y=Y, comment_str=comment_str)

                if i % 100 == 0:
                    elapsed = time.perf_counter() - start_gen
                    eta = elapsed / (i + 1) * (len(generator) - i)
                    print(
                        f"{mode} sample {i}/{len(generator)}, writing to `{tar_writer.ft.name}`, "
                        f"Elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s"
                    )

        print(f"Done with {mode} - Elapsed time: {time.perf_counter() - start_gen:.1f}")

    print("Total time taken: %d" % (time.perf_counter() - start_time))
