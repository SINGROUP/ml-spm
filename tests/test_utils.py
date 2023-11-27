import os
from pathlib import Path

import numpy as np
from torch import nn, optim
import torch


def test_xyz_read_write():
    from mlspm.utils import read_xyzs, write_to_xyz

    xyz_path = "./test.xyx"
    # fmt: off
    xyz = np.array([
        [ 0.0,  0.1,  0.2,  0.3,  1],  # (x, y, z, charge, element)
        [ 0.0, -0.1, -0.2, -0.3,  6],
        [-0.2,  1.5, -2.5,  0.2, 14]]
    )
    # fmt: on

    write_to_xyz(xyz, xyz_path)
    xyz_read = read_xyzs([xyz_path])[0]

    os.remove(xyz_path)

    assert np.allclose(xyz, xyz_read)


def test_checkpoints():
    from mlspm.utils import load_checkpoint, save_checkpoint

    save_dir = Path("test_checkpoints")

    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0 / (1.0 + 1e-5 * b))
    additional_data = {"test_data": 3, "lr_scheduler": lr_scheduler}

    x, y = np.random.rand(2, 1, 10, 10)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    pred = model(x)
    loss = ((y - pred) ** 2).mean()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    print(loss)

    save_checkpoint(model, optimizer, epoch=1, save_dir=save_dir, additional_data=additional_data)

    model_new = nn.Linear(10, 10)
    optimizer_new = optim.Adam(model.parameters())
    lr_scheduler_new = optim.lr_scheduler.LambdaLR(optimizer, lambda b: 1.0 / (1.0 + 1e-5 * b))
    additional_data = {"test_data": 0, "lr_scheduler": lr_scheduler_new}

    load_checkpoint(model_new, optimizer_new, file_name=save_dir / "model_1.pth", additional_data=additional_data)

    assert np.allclose(model.state_dict()["weight"], model_new.state_dict()["weight"])
    assert np.allclose(optimizer.state_dict()["state"][0]["exp_avg"], optimizer_new.state_dict()["state"][0]["exp_avg"])
    assert np.allclose(lr_scheduler.state_dict()["_last_lr"], lr_scheduler_new.state_dict()["_last_lr"])
    assert np.allclose(additional_data["test_data"], 3)
