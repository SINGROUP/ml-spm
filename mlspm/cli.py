import argparse


def _bool_type(value):
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    raise KeyError(f"`{value}` can't be interpreted as a boolean.")


def parse_args() -> dict:
    """
    Parse some useful CLI arguments for use in training scripts.

    Returns:
        A dictionary of the argument values.
    """
    parser = argparse.ArgumentParser("Fit ML model")
    parser.add_argument("--run_dir", type=str, default="./", help="Path to directory where output files are saved. Default = ./")
    parser.add_argument("--dataset", type=str, help="Name of dataset to use.")
    parser.add_argument("--data_dir", type=str, help="Path to directory where the data is stored.")
    parser.add_argument("--urls_train", type=str, help="Webdataset URLs for the training set shards.")
    parser.add_argument("--urls_val", type=str, help="Webdataset URLs for the validation set shards.")
    parser.add_argument("--urls_test", type=str, help="Webdataset URLs for the test set shards.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Number of samples per batch per parallel rank.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument(
        "--classes",
        type=lambda s: [int(n) for n in s.split(",")],
        nargs="+",
        help="Lists of classes for categorizing chemical elements. Elements in each list are separated by commas, and lists are separated by spaces.",
    )
    parser.add_argument("--class_colors", type=str, nargs="+", help="Colors for each class of chemical elements.")
    parser.add_argument(
        "--z_lims", type=float, nargs=2, help="Lower and upper limit for position distribution grid in the z direction."
    )
    parser.add_argument(
        "--box_res", type=float, nargs=3, help="Voxel resolution in the position distribution grid in each direction in Ångströms."
    )
    parser.add_argument("--edge_cutoff", type=float, help="Edge cutoff distance in Ångströms.")
    parser.add_argument("--afm_cutoff", type=float, help="AFM region cutoff around each atom position in Ångströms.")
    parser.add_argument("--zmin", type=float, help="Lowest atom z-coordinate to include in Ångströms (top atom is at 0).")
    parser.add_argument("--peak_std", type=float, help="Position distribution grid peak standard deviation in Ångströms.")
    parser.add_argument(
        "--loss_weights", type=float, nargs="+", default=[1.0], help="Weights for each loss component. Default = [1.0]"
    )
    parser.add_argument(
        "--loss_labels", type=str, nargs="+", default=["MSE"], help='Labels for each loss component. Default = ["MSE"]'
    )
    parser.add_argument(
        "--load_weights", type=str, default="", help="Path to a saved model state to load in the beginning of the training."
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Set random seed for reproducibility. Default = 0.")
    parser.add_argument("--train", type=_bool_type, default=True, help="Enable training (true of false). Default = true.")
    parser.add_argument("--test", type=_bool_type, default=True, help="Enable testing (true of false). Default = true.")
    parser.add_argument("--predict", type=_bool_type, default=True, help="Enable prediction (true of false). Default = true.")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of parallel workers for data loading per parallel rank. Default = 1."
    )
    parser.add_argument(
        "--print_interval", type=int, default=10, help="Number of batches between printing training loss etc. Default = 10."
    )
    parser.add_argument("--comm_backend", type=str, default="nccl", help="Parallel communications backend. Default = nccl.")
    parser.add_argument("--timings", action="store_true", help="Enable printing timings during training.")
    parser.add_argument("--pred_batches", type=int, default=3, help="Number of prediction batches. Default = 3.")
    parser.add_argument(
        "--avg_best_epochs", type=int, default=3, help="Number of epochs to average the best validation loss over. Default = 3."
    )
    args = parser.parse_args()
    return vars(args)
