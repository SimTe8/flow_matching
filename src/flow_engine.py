import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import sample_gaussian


def compute_conditional_flow(x1, device="cuda"):
    """Compute interpolations and targets using your custom sampler."""
    batch_size = x1.shape[0]

    # 1. Use your modular sampler for p0
    x0 = sample_gaussian(batch_size, dim=x1.shape[1], device=device)

    # 2. Sample time steps t uniformly in [0, 1]
    t = torch.rand(batch_size, 1, device=device)

    # 3. Linear interpolation (path x_t)
    t_expanded = t.view(
        batch_size, *([1] * (len(x1.shape) - 1))
    )  # change shape / broadcasting
    xt = t_expanded * x1 + (1.0 - t_expanded) * x0

    # 4. Conditional vector field target (u_t)
    ut = x1 - x0

    return xt, t, ut


def cfm_loss(model, x1, device="cuda"):
    """Berechnet den mittleren quadratischen Fehler (MSE) zwischen vorhergesagtem

    und wahrem Vektorfeld.
    """
    xt, t, ut = compute_conditional_flow(x1, device)

    # the model tries to predict the vector field ut based on the position xt and time t
    vt = model(xt, t)

    # MSE Loss
    loss = torch.mean((vt - ut) ** 2)
    return loss


def train_step(model, optimizer, x1, loss_fn=cfm_loss, device="cuda"):
    """Performs a single training step: computes loss, backpropagates, and updates model parameters."""
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model, x1, device)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_loop(
    model,
    optimizer,
    x1,
    n_epochs=1000,
    device="cuda",
    save_model=True,
    save_bestmodel=False,
    save_last_checkpoint=False,
    save_loss=True,
    plot_loss=True,
    folderpath="results",
):
    """Trains the model for a specified number of epochs."""
    loss_history = []
    for epoch in range(n_epochs):
        loss = train_step(model, optimizer, x1, device=device)
        loss_history.append(loss)
        if epoch == 0:
            print(f"Initial Loss: {loss:.4f}")
            min_loss = loss

        if loss < min_loss:
            min_loss = loss
            if save_bestmodel:
                store_model(model, folderpath=folderpath, filename="model_best")
        elif (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}")

    if save_model:
        store_model(model, folderpath=folderpath, filename=None)
    if save_last_checkpoint:
        store_checkpoint(
            model,
            optimizer=optimizer,
            epoch=n_epochs,
            folderpath=folderpath,
            filename=None,
        )
    if save_loss:
        loss_path = os.path.join(folderpath, "loss_history")
        # save as pytorch file for later loading
        torch.save(loss_history, f"{loss_path}.pth")
        # csv with numpy
        loss_np = np.array(loss_history)
        np.savetxt(f"{loss_path}.csv", loss_np, fmt="%.6f")

        print(f"Loss history saved to {folderpath}")

    if plot_loss:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(loss_history)
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        fig.savefig(
            os.path.join(folderpath, "loss_history.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        os.startfile(os.path.join(folderpath, "loss_history.png"))

    return loss_history


def store_model(model, folderpath="results", filename=None):
    """Saves the model's state dictionary to a file."""

    daystamp = datetime.now().strftime("%m-%d-%y")

    if filename is None:
        filename = f"model_{daystamp}.pth"
    else:
        filename = f"{filename}_{daystamp}.pth"

    path = os.path.join(folderpath, filename)

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def store_checkpoint(
    model, optimizer=None, epoch=None, folderpath="results", filename=None
):
    """Saves the model and optimizer state dictionaries to a file."""

    daystamp = datetime.now().strftime("%m-%d-%y")

    if filename is None:
        filename = f"model_{daystamp}.pth"
    else:
        filename = f"{filename}_{daystamp}.pth"

    path = os.path.join(folderpath, filename)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Loads the model and optimizer state dictionaries from a file."""
    checkpoint = torch.load(path)

    # recover model and optimizer state
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # read out epoch
    start_epoch = checkpoint["epoch"]
    return model, optimizer, start_epoch
