The scripts here can be used for training the position prediction and graph construction models as well as evaluating the trained models.
- `fit_posnet.py`: Train the `PosNet` model that predicts atom positions from AFM images.
- `fit_graphnet.py`: Train the `GraphImgNet` model that constructs a molecule graph out of the predicted atom positions and the AFM image.
- `run_fit_posnet.sh`: Calls the `fit_posnet.py` script with the parameters used for training the models in the paper.
- `run_fit_graphnet.sh`: Calls the `fit_graphnet.py` script with the parameters used for training the models in the paper.
- `test_graphnet.py`: Combines the trained models and runs a test on them.

Run `python fit_posnet.py --help` to get short explanations of all the argument options.

The training scripts for the two models can be run independent of one another, but the testing script can only be run after both of the models have been trained.

Note that running the training scripts triggers the downloading of the training dataset which is a few GB in size.
