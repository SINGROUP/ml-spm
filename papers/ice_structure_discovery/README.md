# Structure discovery in Atomic Force Microscopy imaging of ice

Paper: https://arxiv.org/abs/2310.17161 (Under review)

Abstract: _The interaction of water with surfaces is crucially important in a wide range of natural and technological settings. In particular, at low temperatures, unveiling the atomistic structure of adsorbed water clusters would provide valuable data for understanding the ice nucleation process. Using high-resolution Atomic Force Microscopy (AFM) and Scanning Tunneling Microscopy, several studies have demonstrated the presence of water pentamers, hexamers, heptamers (and of their combinations) on a variety of metallic surfaces, as well the initial stages of 2D ice growth on an insulating surface. However, in all these cases, the observed structures were completely flat, providing a relatively straightforward path to interpretation. Here, we present high-resolution AFM measurements of several new water clusters on Au(111) and Cu(111), whose understanding presents significant challenges, due to both their highly 3D configuration and to their large size. For each of them, we use a combination of machine learning, atomistic modelling with neural network potentials and statistical sampling to propose an underlying atomic structure, finally comparing its AFM simulated images to the experimental ones. These results provide new insights into the early phases of ice formation, which is a ubiquitous phenomenon ranging from biology to astrophysics._

![Workflow](workflow.png)

## Scripts

The subdirectories contain various scripts for training and running predictions with the models:
- `training`: Scripts for training the atom position and graph construction models, and evaluating the trained models.
- `prediction`: Scripts for reproducing the result in Fig. 2 of the paper using the pretrained models.

## Data

Various datasets were used for achieving the results in the paper. This is a listing of those datasets. They will also be automatically downloaded when running the training or prediction scripts above.

Training datasets:
- Cu(111): https://doi.org/10.5281/zenodo.10047850
- Au(111), monolayer: https://doi.org/10.5281/zenodo.10049832
- Au(111), bilayer: https://doi.org/10.5281/zenodo.10049856

Experimental data: https://doi.org/10.5281/zenodo.10054847

Pretrained weights for the models: https://doi.org/10.5281/zenodo.10054348
