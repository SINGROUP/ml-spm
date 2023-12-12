import gc
import os
from pathlib import Path

import numpy as np
import torch

import mlspm.preprocessing as pp
from mlspm import graph, utils
from mlspm.models import GraphImgNetIce
from mlspm.datasets import download_dataset

MM_TO_INCH = 1 / 25.4


def load_data(data_path: os.PathLike):

    # Load data
    data = np.load(data_path)
    X = data["data"][None]
    afm_dim = (data["lengthX"], data["lengthY"])

    # Preprocess
    X = apply_preprocessing([X], afm_dim)

    return X


def make_prediction(model: GraphImgNetIce, X, match_threshold, device):
    with torch.no_grad():
        box_borders = graph.make_box_borders(X.shape[1:3], (model.afm_res, model.afm_res), z_range=model.posnet.grid_z_range)
        xt = torch.from_numpy(X).float().to(device)
        xt, _ = model.posnet(xt.unsqueeze(1))
        pos, matches, labels = graph.find_gaussian_peaks(
            xt, box_borders, match_threshold=match_threshold, std=model.posnet.peak_std, method=model.posnet.match_method
        )
        pred_graph, _ = model.predict_graph(X, pos=pos)
        pred_grid = xt.cpu().numpy()
        matches = matches.cpu().numpy()
        labels = labels.cpu().numpy()
    return pred_graph, pred_grid, matches, labels, box_borders


def apply_preprocessing(X, real_dim):
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = X[0]
    return X


if __name__ == "__main__":
    
    exp_data_dir = Path("./exp_data")
    match_thresholds = {
        "cu111": 0.5,
        "au111-monolayer": 0.5,
        "au111-bilayer": 0.6,
    }
    classes = [[1], [8]]
    device = "cuda"

    exp_data_files = [
        "Chen_CO.npz",
        "Ying_Jiang_1.npz",
        "Ying_Jiang_2_1.npz",
        "Ying_Jiang_2_2.npz",
        "Ying_Jiang_3.npz",
        # "Ying_Jiang_4.npz", # Takes a lot of video memory
        "Ying_Jiang_5.npz",
        "Ying_Jiang_6.npz",
        "Ying_Jiang_7.npz",
    ]

    # Download experimental dataset
    download_dataset("AFM-ice-exp", exp_data_dir)

    for weights in ["cu111", "au111-monolayer", "au111-bilayer"]:

        model = GraphImgNetIce(pretrained_weights=weights, device=device)

        out_dir = Path(f"predictions_{weights}")
        out_dir.mkdir(exist_ok=True)

        print(f"Model: {weights}")

        for exp_data_file in exp_data_files:

            save_name = Path(exp_data_file).stem
            print(f"Experiment: {save_name}")

            # Load data and run prediction
            X = load_data(exp_data_dir / exp_data_file)
            pred_graph, pred_grid, matches, labels, box_borders = make_prediction(model, X, match_thresholds[weights], device=device)

            # Construct xyz array from the graph
            xyzs = np.concatenate(
                [
                    pred_graph[0].array(xyz=True),
                    # Take the elements from the first entry of the element list for the predicted class
                    np.array([classes[ind][0] for ind in pred_graph[0].array(class_index=True)[:, 0]])[:, None],
                ],
                axis=1,
            )

            # Save atom positions
            utils.write_to_xyz(xyzs, outfile=out_dir / f"{save_name}_mol.xyz", verbose=0)

            # Save bond information
            with open(out_dir / f"{save_name}_bonds.txt", "w") as f:
                for b in pred_graph[0].bonds:
                    f.write(f"{b[0]} {b[1]}\n")

        # Minimize memory usage
        del model
        gc.collect()
        torch.cuda.empty_cache()
