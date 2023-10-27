import argparse


def _bool_type(value):
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    raise KeyError(f"`{value}` can't be interpreted as a boolean.")


def parse_args() -> dict:
    # TODO add help
    parser = argparse.ArgumentParser("Fit ML model")
    parser.add_argument("--run_dir", type=str, default="./")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--urls_train", type=str)
    parser.add_argument("--urls_val", type=str)
    parser.add_argument("--urls_test", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--classes", type=lambda s: [int(n) for n in s.split(",")], nargs="+")
    parser.add_argument("--class_colors", type=str, nargs="+")
    parser.add_argument("--z_lims", type=float, nargs=2)
    parser.add_argument("--box_res", type=float, nargs=3)
    parser.add_argument("--edge_cutoff", type=float)
    parser.add_argument("--afm_cutoff", type=float)
    parser.add_argument("--zmin", type=float)
    parser.add_argument("--peak_std", type=float)
    parser.add_argument("--loss_weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--loss_labels", type=str, nargs="+", default=["MSE"])
    parser.add_argument("--load_weights", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--train", type=_bool_type, default=True)
    parser.add_argument("--test", type=_bool_type, default=True)
    parser.add_argument("--predict", type=_bool_type, default=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--comm_backend", type=str, default="nccl")
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--pred_batches", type=int, default=3)
    parser.add_argument("--avg_best_epochs", type=int, default=3)
    args = parser.parse_args()
    return vars(args)
