# Simple example to use flow matching on a 2D toy dataset. You can run this file directly to see how the training and integration work.
import os

import torch

from src.data import (
    plot_samples,
    sample_circle,
    sample_gaussian,
    sample_letters,
    sample_moons,
)
from src.flow_engine import train_loop
from src.models.models import MLPVectorField, SimpleMLPVectorField
from src.solver import animate_flow, integrate
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
    "experiment_name": "moons_no_t_dep",
    "dataset": "moons",  # options: "moons", "circles", "letters"
    "letters": "ST",  # only used if dataset == "letters"
    "n_train_samples": 10000,
    "n_test_samples": 200,
    # model settings
    "model": "SimpleMLP_VF",  # options: "SimpleMLP_VF", "MLP_VF"
    "hidden_dim": 128,
    "optimizer": "adam",
    "lr": 1e-3,
    # integration settings
    "integrator": "rk4",
    "h": 1e-2,
    "n_steps": 100,
    "noise_level": 0.03,
    # training settings
    "train_config": {
        "n_epochs": 10000,
        "device": device,
        "save_model": True,
        "save_bestmodel": False,
        "save_last_checkpoint": False,
        "save_loss": True,
        "plot_loss": True,
    },
    "animate": True,
    "save_animation": True,
}

# setup directory and save config
run_dir = setup_run_dir(config["experiment_name"])
save_config(run_dir, config)


# define dataset
match config["dataset"]:
    case "moons":
        x1 = sample_moons(
            config["n_train_samples"], device=device, noise=config["noise_level"]
        )
    case "circles":
        x1 = sample_circle(
            config["n_train_samples"],
            device=device,
            radius=0.5,
            noise=config["noise_level"],
        )
    case "letters":
        x1 = sample_letters(
            config["n_train_samples"],
            letters=config["letters"],
            device=device,
            noise=config["noise_level"],
        )
    case _:
        raise ValueError(
            f"Unknown dataset: {config['dataset']}. Please choose from 'moons', 'circles', or 'letters'."
        )

# define model
match config["model"]:
    case "SimpleMLP_VF":
        model = SimpleMLPVectorField().to(device)
    case "MLP_VF":
        model = MLPVectorField().to(device)
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
loss_history = train_loop(
    model, optimizer, x1, folderpath=run_dir, **config["train_config"]
)
print("Training completed.\n")

# integration
x0 = sample_gaussian(config["n_test_samples"], device=device)
x_final, trajectories = integrate(
    model, x0, n_steps=config["n_steps"], h=config["h"], integrator=config["integrator"]
)

plot_samples(
    x_final,
    "Final Samples x1 (after integration)",
    save=os.path.join(run_dir, "final_samples.png"),
    show=True,
)

# animation
if config["animate"]:
    print("Generating animation...\n")
    ani = animate_flow(trajectories)
    # ani.save(
    #     os.path.join(run_dir, "flow_matching_animation.gif"), writer="pillow", fps=20
    # )
    # plt.show()

    if config["save_animation"]:
        # save gif
        ani.save(
            os.path.join(run_dir, "flow_matching_animation.gif"),
            writer="pillow",
            fps=20,
            dpi=200,
        )
        print(f"Saved animation in {run_dir}!")
        # # save mp4 (requires ffmpeg installed)
        # ani.save(
        #     os.path.join(run_dir, "flow_matching_animation.mp4"),
        #     writer="ffmpeg",
        #     fps=20,
        #     dpi=200,
        # )
