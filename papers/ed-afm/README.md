# Electrostatic Discovery Atomic Force Microscopy

The scrpts in the original repository at https://github.com/SINGROUP/ED-AFM are reproduced here for convenience.

Paper: [*N. Oinonen et al. Electrostatic Discovery Atomic Force Microscopy, ACS Nano 2022*](https://pubs.acs.org/doi/10.1021/acsnano.1c06840)

Abstract:
_While offering unprecedented resolution of atomic and electronic structure, Scanning Probe Microscopy techniques have found greater challenges in providing reliable electrostatic characterization at the same scale. In this work, we introduce Electrostatic Discovery Atomic Force Microscopy, a machine learning based method which provides immediate quantitative maps of the electrostatic potential directly from Atomic Force Microscopy images with functionalized tips. We apply this to characterize the electrostatic properties of a variety of molecular systems and compare directly to reference simulations, demonstrating good agreement. This approach opens the door to reliable atomic scale electrostatic maps on any system with minimal computational overhead._

![Method schematic](https://github.com/SINGROUP/ED-AFM/blob/master/figures/method_schem.png)

## ML model

We use a U-net type convolutional neural network with attention gates in the skip connections. Similar model was used previously by Oktay et al. for segmenting medical images (https://arxiv.org/abs/1804.03999v2).

![Model schematic](https://github.com/SINGROUP/ED-AFM/blob/master/figures/model_schem.png)
![AG schematic](https://github.com/SINGROUP/ED-AFM/blob/master/figures/AG_schem.png)

The model implementation can be found in [`mlspm/image/models.py`](https://github.com/SINGROUP/ml-spm/blob/main/mlspm/image/models.py), where two modules can be found: [`AttentionUNet`](https://ml-spm.readthedocs.io/en/latest/reference/mlspm.image.html#mlspm.image.models.AttentionUNet), which is the generic version of the model, and [`EDAFMNet`](https://ml-spm.readthedocs.io/en/latest/reference/mlspm.image.html#mlspm.image.models.EDAFMNet), which is a subclass of the former specifying the exact hyperparameters for the model that we used.

In `EDAFMNet` one can also specify pretrained weights of several types to download to the model using the `pretrained_weights` argument:

 - `'base'`: The base model used for all predictions in the main ED-AFM paper and used for comparison in the various test in the supplementary information of the paper.
 - `'single-channel'`: Model trained on only a single CO-tip AFM input.
 - `'CO-Cl'`: Model trained on alternative tip combination of CO and Cl.
 - `'Xe-Cl`': Model trained on alternative tip combination of Xe and Cl.
 - `'constant-noise'`: Model trained using constant noise amplitude instead of normally distributed amplitude.
 - `'uniform-noise'`: Model trained using uniform random noise amplitude instead of normally distributed amplitude.
 - `'no-gradient'`: Model trained without background-gradient augmentation.
 - `'matched-tips'`: Model trained on data with matched tip distance between CO and Xe, instead of independently randomized distances.

The model weights can also be downloaded directly from https://doi.org/10.5281/zenodo.10606273. The weights are saved in the state_dict format of PyTorch.

## Data and model training

We provide a database of molecular geometries that can be used to generate the full dataset using [`ppafm`](https://github.com/Probe-Particle/ppafm). The provided script `generate_data.py` does the data generation and will download the molecule database automatically. Alternatively, the molecule database can be downloaded directly from https://doi.org/10.5281/zenodo.10609676.

After generating the dataset, the model training can be done using the provided script `run_train.sh`, which calls the actual training script `train.py` with the appropriate parameters. Note that performing the training using all the same settings as we used requires a significant amount of time and also a significant amount VRAM on the GPU, likely more than can be found on a single GPU. In our case the model training took ~5 days using 4 x Nvidia Tesla V100 (32GB) GPUs. However, inference on the trained model can be done even on a single lower-end GPU or on CPU.

All the data used for the predictions in the paper can be found at https://doi.org/10.5281/zenodo.10609676.

## Figures

The scripts used to generate most of the figures in the paper are provided under the directory `figures`. Running the scripts will automatically download the required data.

The scripts correspond to the figures as follows:

 - Fig. 1: `sims.py`
 - Fig. 2: `ptcda.py`
 - Fig. 3: `bcb.py`
 - Fig. 4: `water.py`
 - Fig. 5: `surface_sims_bcb_water.py`
 - Fig. S1: `model_schem.tex`
 - Fig. S3: `stats.py`\*
 - Fig. S4: `esmap_sample.py` and then `esmap_schem.tex`
 - Fig. S5: `stats_spring_constants.py`\*
 - Fig. S6: `afm_stacks.py` and `afm_stacks2.py`
 - Fig. S7: `sims_hartree.py`
 - Fig. S8: `ptcda_surface_sim.py`
 - Fig. S9: `single_tip.py`
 - Fig. S10: `sims_Cl.py`
 - Fig. S11: `height_dependence.py`
 - Fig. S12: `extra_electron.py`
 - Fig. S13: `background_gradient.py`

\* Precalculated MSE values used by the plotting script are provided under `figures/stats`. The scripts used to calculate these values are also under `figures/stats`.

You can also use `run_all.sh`-script to run all of the scripts in one go. Note that compiling the .tex files additionally requires a working LaTex installation on your system.
