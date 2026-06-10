# Simple example to use flow matching on a 2D toy dataset. You can run this file directly to see how the training and integration work.

import torch

from src.data import (
    get_mnist_dataloader,
    get_single_digit_mnist_dataloader,
)
from src.flow_engine import train_loop_MNIST
from src.models.models import MiniUNetVectorField
from src.utils import save_config, setup_run_dir

# device settings
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU for training and integration.\n")
else:
    device = "cpu"
    print("Using CPU for training and integration.\n")

# manual override
# device = 'cpu'


# GLOBAL SETTINGS
config = {
    "experiment_name": "MNIST_ext_baseline",
    "dataset": "MNIST",
    "single_digit": None,
    "batch_size": 128,
    "n_test_samples": 10,
    # model settings
    "model": "MiniUNetVectorField",
    "hidden_dim": 32,
    "optimizer": "adam",
    "lr": 1e-3,
    # integration settings
    "integrator": "rk4",
    "h": 1e-1,
    "n_steps": 10,
    # training settings
    "train_config": {
        "n_epochs": 100,
        "device": device,
        "save_model": True,
        "save_bestmodel": True,
        "save_last_checkpoint": False,
        "save_loss": True,
        "plot_loss": True,
    },
    "animate": False,
    "save_animation": False,
}

# setup directory and save config
run_dir = setup_run_dir(config["experiment_name"])
save_config(run_dir, config)


# define dataset
if config["single_digit"] is None:
    dl_train = get_mnist_dataloader(batch_size=config["batch_size"])
else:
    dl_train = get_single_digit_mnist_dataloader(
        digit=config["single_digit"], batch_size=config["batch_size"]
    )


# define model
match config["model"]:
    case "MiniUNetVectorField":
        model = MiniUNetVectorField().to(device)
    case _:
        raise ValueError("Unknown model type.")

# define optimizer
match config["optimizer"]:
    case "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    case "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    case _:
        raise ValueError("Unknown optimizer type.")


# training
print("Starting training...\n")
loss_history = train_loop_MNIST(
    model, optimizer, dl_train, folderpath=run_dir, **config["train_config"]
)
print("Training completed.\n")

# # integration
# x0 = sample_gaussian(config["n_test_samples"], device=device)
# x_final, trajectories = integrate(
#     model, x0, n_steps=config["n_steps"], h=config["h"], integrator=config["integrator"]
# )


# # animation
# if config["animate"]:
#     print("Generating animation...\n")
#     ani = animate_flow(trajectories)
#     # ani.save(
#     #     os.path.join(run_dir, "flow_matching_animation.gif"), writer="pillow", fps=20
#     # )
#     # plt.show()

#     if config["save_animation"]:
#         # save gif
#         ani.save(
#             os.path.join(run_dir, "flow_matching_animation.gif"),
#             writer="pillow",
#             fps=20,
#             dpi=200,
#         )
#         print(f"Saved animation in {run_dir}!")
#         # # save mp4 (requires ffmpeg installed)
#         # ani.save(
#         #     os.path.join(run_dir, "flow_matching_animation.mp4"),
#         #     writer="ffmpeg",
#         #     fps=20,
#         #     dpi=200,
#         # )
