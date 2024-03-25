from math import ceil
import time
from pathlib import Path
from mlspm.datasets import download_dataset

from ppafm.common import sphereTangentSpace
from ppafm.ml.AuxMap import AtomicDisks, HeightMap, vdwSpheres
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl.AFMulator import AFMulator

from mlspm.data_generation import TarWriter


class Trainer(InverseAFMtrainer):
    # Override to randomize tip distance and probe tilt
    def on_sample_start(self):
        self.randomize_distance(delta=0.25)
        self.randomize_tip(max_tilt=0.5)


if __name__ == "__main__":
    # Which dataset to generate ("light" or "heavy")
    dataset = "heavy"

    # Path where molecule geometry files are saved
    mol_dir = Path("./molecules")

    # Directory where to save data
    data_dir = Path(f"./data_{dataset}/")

    # Define simulator and image descriptor parameters
    scan_window = ((0, 0, 6.0), (15.875, 15.875, 7.9))
    scan_dim = (128, 128, 19)
    afmulator = AFMulator(pixPerAngstrome=5, scan_dim=scan_dim, scan_window=scan_window, tipR0=[0.0, 0.0, 4.0])
    aux_maps = [
        AtomicDisks(scan_dim=scan_dim, scan_window=scan_window, zmin=-1.2, zmax_s=-1.2, diskMode="sphere"),
        vdwSpheres(scan_dim=scan_dim, scan_window=scan_window, zmin=-1.5),
        HeightMap(scanner=afmulator.scanner, zmin=-2.0),
    ]
    generator_arguments = {
        "afmulator": afmulator,
        "aux_maps": aux_maps,
        "batch_size": 1,
        "distAbove": 5.25,
        "iZPPs": [8],
        "Qs": [[-0.1, 0, 0, 0]],
        "QZs": [[0, 0, 0, 0]],
    }
    rotations = sphereTangentSpace(n=100)

    # Number of tar file shards for each set
    target_shard_count = 16

    if dataset == "light":
        N_train = 5000  # Number of training molecules from light molecule set
        N_val = 1500  # Number of validation molecules from light molecule set
        N_test = 2500  # Number of test molecules from light molecule set
        N_train_h = 0  # Number of training molecules from heavy molecule set
        N_val_h = 0  # Number of validation molecules from heavy molecule set
        N_test_h = 0  # Number of test molecules from heavy molecule set
    elif dataset == "heavy":
        N_train = 3500
        N_val = 900
        N_test = 1200
        N_train_h = 2500
        N_val_h = 600
        N_test_h = 1200
    else:
        raise ValueError(f"Invalid dataset `{dataset}`")

    # Heavy molecules
    train_molecules = [mol_dir / f"heavy/{n}.xyz" for n in range(N_train_h)]
    val_molecules = [mol_dir / f"heavy/{n}.xyz" for n in range(N_train_h, N_train_h + N_val_h)]
    test_molecules = [mol_dir / f"heavy/{n}.xyz" for n in range(N_train_h + N_val_h, N_train_h + N_val_h + N_test_h)]

    # Light molecules
    train_molecules += [mol_dir / f"light/{n}.xyz" for n in range(N_train)]
    val_molecules += [mol_dir / f"light/{n}.xyz" for n in range(N_train, N_train + N_val)]
    test_molecules += [mol_dir / f"light/{n}.xyz" for n in range(N_train + N_val, N_train + N_val + N_test)]

    # Make sure the save directory exists
    data_dir.mkdir(exist_ok=True, parents=True)

    # Download the dataset. The extraction may take a while since there are ~140k files.
    download_dataset("ASD-AFM-molecules", mol_dir)

    # Generate dataset
    start_time = time.perf_counter()
    counter = 1
    total_len = len(train_molecules) + len(val_molecules) + len(test_molecules)
    for mode, molecules in zip(["train", "val", "test"], [train_molecules, val_molecules, test_molecules]):
        # Construct generator
        generator = Trainer(paths=molecules, **generator_arguments)
        generator.augment_with_rotations_entropy(rotations, n_best_rotations=30)

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
                    print(f"{mode} sample {i}/{len(generator)}, Elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s")

        print(f"Done with {mode} - Elapsed time: {time.perf_counter() - start_gen:.1f}")

    print("Total time taken: %d" % (time.perf_counter() - start_time))
