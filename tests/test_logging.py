
import os
import numpy as np

def test_loss_log_plot():
    from mlspm.logging import LossLogPlot

    loss_log_path = "loss_log.csv"
    plot_path = "test_plot.png"
    log_path = "test_log.txt"

    info_log = open(log_path, 'w')

    if os.path.exists(loss_log_path):
        os.remove(loss_log_path)
    loss_log = LossLogPlot(loss_log_path, plot_path, loss_labels=["1", "2"], loss_weights=["", ""], stream=info_log)

    losses = [[[0, 1, 2], [1, 2, 3]], [[1.5, 2.0, 3.0], [0.1, 0.4, 0.7]]]
    for loss in losses:
        loss_log.add_train_loss(loss[0])
        loss_log.add_val_loss(loss[1])
    loss_log.next_epoch()

    new_log = LossLogPlot(loss_log_path, "plot.png", loss_labels=["1", "2"], loss_weights=["", ""], stream=info_log)

    info_log.close()

    os.remove(loss_log_path)
    os.remove(plot_path)
    os.remove(log_path)

    assert new_log.epoch == 2
    assert np.allclose(new_log.train_losses, np.array([[0.75, 1.5, 2.5]])), new_log.train_losses
    assert np.allclose(new_log.val_losses, np.array([[0.55, 1.2, 1.85]])), new_log.val_losses
