# Create data sampling functions for source and target distributions
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import torch
from PIL import Image, ImageDraw, ImageFont


# 1. Source distribution p0
def sample_gaussian(batch_size, dim=2, device="cuda"):
    """Sample from standard normal distribution p0."""
    return torch.randn(batch_size, dim, device=device)


# 2. Target distribution p1
def sample_moons(batch_size, device="cuda", noise=0.0):
    """Sample from Moons distribution p1, normalized and moved to device."""
    data, _ = sklearn.datasets.make_moons(n_samples=batch_size, noise=noise)

    # Convert to tensor and center/scale data for smoother training
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = (data - data.mean(dim=0)) / data.std(dim=0)

    return data


def sample_circle(batch_size, device="cuda", radius=0.5, noise=0.0):
    """Sample from a circle distribution."""

    data, _ = sklearn.datasets.make_circles(
        n_samples=batch_size, noise=noise, factor=radius
    )

    # Convert to tensor and center/scale data for smoother training
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    return data


def sample_letters(batch_size, letters="HD", device="cuda", noise=0.03):
    """Generate 2D samples forming the given letters, e.g. 'HD'."""
    # 1. Create a black and white image with text
    w, h = 200, 100
    img = Image.new("L", (w, h), color=0)
    canvas = ImageDraw.Draw(img)

    # Use default font (or specify a ttf path for cleaner look)
    try:
        font = ImageFont.load_default(size=60)
    except TypeError:
        font = ImageFont.load_default()  # Fallback for older PIL versions

    # Draw text centered
    canvas.text((25, 10), letters, fill=255, font=font)

    # 2. Get coordinates of bright pixels
    img_array = np.array(img)
    y_idx, x_idx = np.where(img_array > 128)

    # 3. Normalize coordinates to be centered around 0 with reasonable variance
    x_data = x_idx.astype(np.float32)
    y_data = -y_idx.astype(np.float32)  # Invert y to match standard cartesian grid

    x_data = (x_data - x_data.mean()) / x_data.std()
    y_data = (y_data - y_data.mean()) / y_data.std()
    coords = np.stack([x_data, y_data], axis=1)

    # 4. Sample a random batch from these coordinates and add light noise
    random_indices = np.random.choice(coords.shape[0], size=batch_size)
    batch_data = coords[random_indices]

    # Convert to tensor and add small Gaussian noise for density
    tensor_data = torch.tensor(batch_data, dtype=torch.float32, device=device)
    tensor_data += torch.randn_like(tensor_data) * noise

    return tensor_data


def plot_samples(samples, title, save=None, show=False):
    """Utility function to plot samples."""
    plt.figure(figsize=(5, 4))
    # pay attention to only plot on cpu
    plt.scatter(
        samples[:, 0].cpu().detach().numpy(),
        samples[:, 1].cpu().detach().numpy(),
        alpha=0.5,
    )
    plt.title(title)
    plt.axis("equal")
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.close()
        if show:
            os.startfile(save)
    # plt.show()
