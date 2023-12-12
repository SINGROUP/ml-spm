The scripts here can be used to reproduce the results figures in the paper.
- `predict_experiments.py`: Runs the prediction for all of the experimental AFM images of ice on Cu(111) and Au(111) using the three models pretrained on the Cu(111), Au(111)-monolayer, and Au(111)-bilayer datasets, and saves them on disk.
- `plot_predictions.py`: Picks the appropriate predictions for each experiment and plots them to a figure as in Fig. 2 of the paper.
- `plot_relaxed_structures.py`: Plots the on-surface structures relaxed with a neural network potential and DFT as well as the corresponding simulations and experimental images as in Fig. 3 of the paper.
- `plot_prediction_extra.py`: Plots the prediction and the relaxed structure with corresponding simulations and experimental images for the one extra ice cluster not in the main results figure.
