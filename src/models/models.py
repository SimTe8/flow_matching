import torch
import torch.nn as nn


class SimpleMLPVectorField(nn.Module):
    """Standard time-conditioned MLP for 2D vector fields."""

    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()

        self.name = "SimpleMLP_VF"

        # Input is data_dim (2) + time_dim (1) = 3
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, xt, t):
        # Concatenate x_t and t along the feature dimension
        x_input = torch.cat([xt, t], dim=-1)
        return self.net(x_input)


# more elaborate model
class MLPVectorField(nn.Module):
    """Time-dependent MLP vector field for 2D Flow Matching."""

    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()

        self.name = "MLP_VF"

        # Time embedding layer to project scalar t to hidden space
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Spatial feature layers
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),  # Output dim equals data dim (2D)
        )

    def forward(self, xt, t):
        # xt shape: (batch_size, 2)
        # t shape: (batch_size, 1)

        # 1. Embed time
        t_embed = self.time_mlp(t)  # (batch_size, hidden_dim)

        # 2. Process spatial features and modulate with time
        # We can pass through layers and add time embedding for conditioning
        x = nn.Linear(xt.shape[-1], t_embed.shape[-1]).to(xt.device)(
            xt
        )  # Initial projection
        x = torch.sin(x) + t_embed  # Simple addition of time context

        # # Alternative: Just concat for the simple 2D case if preferred
        # # Here we use a clean modular approach
        # x_combined = xt + t_embed[:, : xt.shape[-1]]  # Quick baseline shortcut

        # # Let's do the standard ML-way: simple MLP that takes concatenated features
        # return self.net(xt + t_embed[:, : xt.shape[-1]])
