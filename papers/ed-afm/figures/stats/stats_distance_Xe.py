
import random
import time
from pathlib import Path

import numpy as np
import ppafm.ml.AuxMap as aux
import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu
import ppafm.ocl.relax as oclr
import torch
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl.AFMulator import AFMulator

import mlspm.preprocessing as pp
from mlspm.datasets import download_dataset
from mlspm.models import EDAFMNet


class Trainer(InverseAFMtrainer):

    # Override this method to set the Xe tip at a different height
    def handle_distance(self):
        if self.afmulator.iZPP == 54:
            self.distAboveActive = self.distAboveXe
        super().handle_distance()
        if self.afmulator.iZPP == 54:
            self.distAboveActive = self.distAbove

def apply_preprocessing(batch):
    Xs, Ys, _ = batch
    Xs = [x[...,2:8] for x in Xs]
    pp.add_norm(Xs)
    pp.add_noise(Xs, c=0.08, randomize_amplitude=False)
    return Xs, Ys

if __name__ == "__main__":

    # # Independent tips model
    # model_type      = "base"                         # Type of pretrained weights to use
    # save_file       = Path("mse_independent_Xe.csv") # File to save MSE values into

    # Matched tips model
    model_type      = "matched-tips"                # Type of pretrained weights to use
    save_file       = Path("./mse_matched_Xe.csv")  # File to save MSE values into

    device          = "cuda"                        # Device to run inference on
    molecules_dir   = Path("../../molecules")       # Path to molecule database
    test_heights    = np.linspace(4.9, 5.7, 21)     # Test heights to run
    n_samples       = 3000                          # Number of samples to run

    if save_file.exists():
        raise RuntimeError("Save file already exists")

    afmulator_args = {
        "pixPerAngstrome"   : 20,
        "scan_dim"          : (128, 128, 19),
        "scan_window"       : ((2.0, 2.0, 6.0), (18.0, 18.0, 7.9)),
        "df_steps"          : 10,
        "tipR0"             : [0.0, 0.0, 4.0]
    }

    generator_kwargs = {
        "batch_size"    : 30,
        "distAbove"     : 5.3,
        "iZPPs"         : [8, 54],
        "Qs"            : [[ -10, 20,  -10, 0 ], [  30, -60,   30, 0 ]],
        "QZs"           : [[ 0.1,  0, -0.1, 0 ], [ 0.1,   0, -0.1, 0 ]]
    }

    # Set random seed for reproducibility
    random.seed(0)

    # Initialize OpenCL environment on GPU
    env = oclu.OCLEnvironment( i_platform = 0 )
    FFcl.init(env)
    oclr.init(env)

    # Define AFMulator
    afmulator = AFMulator(**afmulator_args)
    afmulator.npbc = (0,0,0)

    # Define AuxMaps
    aux_maps = [
        aux.ESMapConstant(
            scan_dim    = afmulator.scan_dim[:2],
            scan_window = [afmulator.scan_window[0][:2], afmulator.scan_window[1][:2]],
            height      = 4.0,
            vdW_cutoff  = -2.0,
            Rpp         = 1.0
        )
    ]

    # Download molecules if not already there
    download_dataset("ED-AFM-molecules", molecules_dir)

    # Define generator
    xyz_paths = (molecules_dir / "test").glob("*.xyz")
    trainer = Trainer(afmulator, aux_maps, xyz_paths, **generator_kwargs)

    # Pick samples
    random.shuffle(trainer.molecules)
    trainer.molecules = trainer.molecules[:n_samples]

    # Make model
    model = EDAFMNet(device=device, pretrained_weights=model_type)

    # Initialize save file
    with open(save_file, "w") as f:
        pass

    # Calculate MSE at every height for every batch
    start_time = time.time()
    total_len = len(test_heights)*len(trainer)
    for ih, height in enumerate(test_heights):

        print(f"Height = {height:.2f}")
        trainer.distAboveXe = height

        mses = []
        for ib, batch in enumerate(trainer):

            X, ref = apply_preprocessing(batch)
            X = [torch.from_numpy(x).unsqueeze(1).to(device) for x in X]
            ref = [torch.from_numpy(r).to(device) for r in ref]

            with torch.no_grad():
                pred, _ = model(X)
                pred = pred[0]
            
            diff = pred - ref[0]
            for d in diff:
                mses.append((d**2).mean().cpu().numpy())
            
            eta = (time.time() - start_time) * (total_len / (ih*len(trainer)+ib+1) - 1)
            print(f"Batch {ib+1}/{len(trainer)} - ETA: {eta:.1f}s")

        with open(save_file, "a") as f:
            f.write(f"{height:.2f},")
            f.write(",".join([str(v) for v in mses]))
            f.write("\n")
