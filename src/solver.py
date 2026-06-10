import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .data import plot_samples
from .integrators import euler, midpoint, rk2, rk4


def integrate(
    model, x0, n_steps=100, integrator="euler", h=1e-2, plot=False, verbose=True
):

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
        for step in tqdm(range(n_steps), desc="Integrating: ", disable=not verbose):
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


def animate_vf(
    model,
    bounds=(-2, 2),
    grid_res=40,
    n_frames=50,
    device="cuda",
):
    """
    Animates the learned 2D vector field over time and saves it as a GIF.
    """
    model.eval()

    # 1. Create static spatial grid
    x = np.linspace(bounds[0], bounds[1], grid_res)
    y = np.linspace(bounds[0], bounds[1], grid_res)
    X, Y = np.meshgrid(x, y)

    grid_pts = np.stack([X.flatten(), Y.flatten()], axis=1)
    xt_tensor = torch.tensor(grid_pts, dtype=torch.float32).to(device)

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        """Update function for each animation frame."""
        ax.clear()

        # Calculate normalized time t in [0, 1]
        t = frame / max(1, (n_frames - 1))
        t_tensor = torch.full((xt_tensor.shape[0], 1), t, dtype=torch.float32).to(
            device
        )

        # 2. Predict velocities for current t
        with torch.no_grad():
            vt = model(xt_tensor, t_tensor).cpu().numpy()

        U = vt[:, 0].reshape(grid_res, grid_res)
        V = vt[:, 1].reshape(grid_res, grid_res)
        speed = np.sqrt(U**2 + V**2)

        # 3. Draw streamplot
        # ax.streamplot(X, Y, U, V, color=speed, cmap='viridis',
        #               linewidth=1.5, density=1.2, arrowsize=1.5)
        q = ax.quiver(
            X, Y, U, V, speed, cmap="viridis", scale=80, width=0.005, alpha=0.9
        )

        # Format plot
        ax.set_title(f"Time $t = {t:.2f}$", fontsize=16)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_xlabel("$x_1$", fontsize=12)
        ax.set_ylabel("$x_2$", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Simple console progress
        print(f"Rendering frame {frame + 1}/{n_frames}", end="\r")

    print(f"Generating animation ({n_frames} frames)...")

    # Create and save animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    # if save:
    #     writer = animation.PillowWriter(fps=fps)
    #     anim.save(filename, writer=writer)
    #     print(f"\nSuccessfully saved animation to: {filename}")

    plt.close()

    return anim
