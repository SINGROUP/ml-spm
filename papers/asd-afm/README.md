# Automated structure discovery in atomic force microscopy

This is a Pytorch version of Keras/Tensorflow model training code in https://github.com/SINGROUP/ASD-AFM. The original training procedure is not reproduced exactly, so some differences in the training outcomes may be expected.

Paper: [*B. Alldritt et. al, Automated structure discovery in atomic force microscopy, Sci. Adv., 2020.*](https://advances.sciencemag.org/content/6/9/eaay6913.full)

Scripts:
- `generate_data.py`: Download molecule dataset and generate a dataset of AFM images and image descriptors.
- `train.py`: Train and test a model using the generated AFM image dataset.
- `run_train.sh`: Run the training script with preset parameters.
- `predict_1S-camphor.py`: Run prediction for experimental AFM images of 1S-camphor using a pretrained model.

Data:
- https://doi.org/10.5281/zenodo.10562769
    - `afm_camphor.tar.gz`: Experimental AFM images of 1S-camphor on Cu(111)
    - `molecules.tar.gz`: Two datasets of molecules (`light` and `heavy`) with point-charges.
- https://doi.org/10.5281/zenodo.10514470: Pretrained weights for the structure prediction model, one set of weights for each of the molecule datasets (`light` and `heavy`).

