
import pytest

def test_parse_args():
    from mlspm.cli import parse_args

    args = parse_args(["--train", "false", "--predict", "False", '--test', "true", "--classes", "1,2,3", "4,5,6"])

    assert args["train"] == False
    assert args["predict"] == False
    assert args["test"] == True
    assert args["classes"] == [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(KeyError):
        parse_args(["--train", "fals"])
