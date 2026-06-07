import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .data import plot_samples
from .integrators import euler, midpoint, rk2, rk4


def integrate(model, x0, n_steps=100, integrator="euler", h=1e-2, plot=False):

    integrators = {
        "euler": euler,
        "midpoint": midpoint,
        "rk2": rk2,
        "rk4": rk4,
    }

    try:
        solver = integrators[integrator]
    except KeyError:
        raise ValueError(
            f"Unknown integrator: {integrator}. Please choose from {integrators.keys()}."
        )

    if plot:
        plot_samples(x0, "Initial Samples x0")

    def f(xt, t):
        batch_size = xt.shape[0]
        t_array = torch.ones(batch_size, 1, device=xt.device) * t
        model.eval()
        vt = model(xt, t_array)
        return vt

    dt = 1.0 / n_steps
    t = 0.0
    xt = x0
    xt_history = []
    with torch.no_grad():
        for step in tqdm(range(n_steps), desc="Integrating: "):
            t = step * dt
            xt = solver(xt, f, t, h)
            xt_history.append(xt.detach().cpu().numpy())

    if plot:
        plot_samples(xt, "Final Samples x1 (after integration)")
    return xt, np.array(xt_history)


def animate_flow(trajectories):
    """Genaerate a 2D scatter plot animation of the ODE integration process."""
    steps = trajectories.shape[0]
    dt = 1.0 / len(trajectories)

    # 3. Setup matplotlib animation
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(
        trajectories[0][:, 0], trajectories[0][:, 1], alpha=0.6, c="crimson", s=10
    )

    # Keep axis limits stable during animation
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title("Flow Matching Trajectories (t = 0.00)")
    ax.grid(True, linestyle="--", alpha=0.5)

    def update(frame):
        # Update point positions
        scat.set_offsets(trajectories[frame])
        # Update time in title
        current_t = frame * dt
        ax.set_title(f"Flow Matching Trajectories (t = {current_t:.2f})")
        return (scat,)

    # Create animation object
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    plt.close()  # Prevent static plot from showing up

    return ani
