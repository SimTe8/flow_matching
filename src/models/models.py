import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SimpleMLPVectorFieldND(nn.Module):
    """Standard time-conditioned MLP for ND vector fields."""

    def __init__(self, img_size=28, hidden_dim=128):
        super().__init__()

        self.name = "SimpleMLP_VF"
        self.img_size = img_size
        self.flatten = nn.Flatten()
        data_dim = img_size * img_size

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
        xt = self.flatten(xt)
        # Concatenate x_t and t along the feature dimension
        x_input = torch.cat([xt, t], dim=-1)
        return self.net(x_input).reshape(-1, self.img_size, self.img_size)


# more elaborate models with time embedding
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


class ContinuousTimeEmbedding(nn.Module):
    """Transforms scalar t into a high-dimensional frequency vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: (batch_size, 1)
        device = t.device
        half_dim = self.dim // 2

        # Inverted exponents for large frequencies: 1 to 10000
        exponents = torch.linspace(0, math.log(10000), half_dim, device=device)
        frequencies = torch.exp(exponents)
        # frequencies is now a tensor with values between 1 and 10000

        # # Broadcast and compute sin/cos -> sin/cos(t * omega)
        frequencies = t * frequencies[None, :]
        # the frequencies are multiplied with the t values -> for every t value,
        # there are now low and high frequencies
        # now we feed the frequencies into sin and cos, where the high frequencies
        # will oscillate faster. This creates unique fingerprints for every time t
        # and makes them more distinguishable
        embedding = torch.cat((frequencies.sin(), frequencies.cos()), dim=-1)
        return embedding


class TimeEmbeddingMLPVectorFieldND(nn.Module):
    """Time-conditioned MLP with sinusoidal embedding for image vector fields."""

    def __init__(self, img_size=28, hidden_dim=256, time_emb_dim=64):
        super().__init__()

        self.name = "TimeEmbeddingMLP_VF"
        self.img_size = img_size
        data_dim = img_size * img_size

        # 1. High-dimensional time projection
        self.time_emb = ContinuousTimeEmbedding(dim=time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Main network layers
        self.input_layer = nn.Linear(data_dim, hidden_dim)
        self.act = nn.SiLU()

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, data_dim)

    def forward(self, xt, t):
        # xt shape: (batch_size, 1, 28, 28) or (batch_size, 28, 28)
        batch_size = xt.shape[0]
        x = xt.view(batch_size, -1)  # Flatten to (batch_size, 784)

        # Process time condition
        t_features = self.time_emb(t)  # (batch_size, time_dim)
        t_context = self.time_mlp(t_features)  # (batch_size, hidden_dim)

        # Forward pass with time injection at every layer
        h = self.input_layer(x)
        h = self.act(h + t_context)

        h = self.layer1(h)
        h = self.act(h + t_context)

        h = self.layer2(h)
        h = self.act(h + t_context)

        h = self.layer3(h)
        h = self.act(h + t_context)

        out = self.output_layer(h)
        return out.reshape(-1, self.img_size, self.img_size)


# CNN models to test time embedding impact


class CNNVectorFieldNoTime(nn.Module):
    """Pure CNN vector field completely ignoring the time parameter t."""

    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.name = "CNN_NoTime_VF"

        # Identical spatial architecture to the time-conditioned version
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, xt, t):
        # xt shape: (B, 1, 28, 28), t is ignored in computation
        h = F.silu(self.conv1(xt))
        h = F.silu(self.conv2(h))
        out = self.conv3(h)
        return out


class CNNVectorFieldWithTime(nn.Module):
    """CNN vector field with modulated continuous time embedding."""

    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.name = "CNN_WithTime_VF"

        # Time embedding block
        self.time_emb = nn.Sequential(
            ContinuousTimeEmbedding(dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Spatial architecture
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.time_proj1 = nn.Linear(hidden_dim, hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.time_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, xt, t):
        # xt shape: (B, 1, 28, 28), t shape: (B, 1)

        # 1. Compute time contexts
        t_emb = self.time_emb(t)
        t_shift1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        t_shift2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)

        # 2. Forward pass with layer-wise time injection
        h = F.silu(self.conv1(xt) + t_shift1)
        h = F.silu(self.conv2(h) + t_shift2)
        out = self.conv3(h)
        return out


# U-Net


class TimeConvBlock(nn.Module):
    """Convolutional block with time injection."""

    def __init__(self, in_ch, out_ch, time_dim, fake=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        # GroupNorm works better than BatchNorm for generative models
        self.norm = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.fake = fake

    def forward(self, x, t_emb):
        # 1. Project time to match channel dimension
        # Shape: (B, out_ch, 1, 1) for broadcasting over image spatial dims
        t_shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)

        # 2. Convolve and inject time before activation
        h = self.norm(self.conv(x))
        if not self.fake:
            return F.silu(h + t_shift)
        else:
            return F.silu(h)


class MiniUNetVectorField(nn.Module):
    """A small U-Net architecture for 28x28 MNIST Flow Matching."""

    def __init__(self, in_channels=1, base_channels=32, fake=False):
        super().__init__()
        self.name = "MiniUNet_VF"

        time_dim = base_channels * 4
        self.time_emb = nn.Sequential(
            ContinuousTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Downsampling (Encoder)
        # 28x28 -> 14x14
        self.down1 = TimeConvBlock(in_channels, base_channels, time_dim, fake=fake)
        self.pool1 = nn.Conv2d(base_channels, base_channels, 4, 2, 1)

        # 14x14 -> 7x7
        self.down2 = TimeConvBlock(
            base_channels, base_channels * 2, time_dim, fake=fake
        )
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)

        # Bottleneck (7x7)
        self.mid = TimeConvBlock(
            base_channels * 2, base_channels * 2, time_dim, fake=fake
        )

        # Upsampling (Decoder)
        # 7x7 -> 14x14
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        # Skip connection adds base_channels*2 + base_channels*2 = base_channels*4
        self.up_block1 = TimeConvBlock(
            base_channels * 4, base_channels, time_dim, fake=fake
        )

        # 14x14 -> 28x28
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1)
        # Skip connection adds base_channels + base_channels = base_channels*2
        self.up_block2 = TimeConvBlock(
            base_channels * 2, base_channels, time_dim, fake=fake
        )

        # Final output projection
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, xt, t):
        # xt shape: (B, 1, 28, 28)
        t_emb = self.time_emb(t)

        # Encoder path
        d1 = self.down1(xt, t_emb)  # (B, 32, 28, 28)
        p1 = F.silu(self.pool1(d1))  # (B, 32, 14, 14)

        d2 = self.down2(p1, t_emb)  # (B, 64, 14, 14)
        p2 = F.silu(self.pool2(d2))  # (B, 64, 7, 7)

        # Bottleneck
        m = self.mid(p2, t_emb)  # (B, 64, 7, 7)

        # Decoder path with Skip Connections
        u1 = F.silu(self.up1(m))  # (B, 64, 14, 14)
        # Concatenate u1 and d2 along channel dimension
        u1_cat = torch.cat([u1, d2], dim=1)  # (B, 128, 14, 14)
        u1_out = self.up_block1(u1_cat, t_emb)  # (B, 32, 14, 14)

        u2 = F.silu(self.up2(u1_out))  # (B, 32, 28, 28)
        # Concatenate u2 and d1
        u2_cat = torch.cat([u2, d1], dim=1)  # (B, 64, 28, 28)
        u2_out = self.up_block2(u2_cat, t_emb)  # (B, 32, 28, 28)

        # Output vector field
        out = self.out_conv(u2_out)  # (B, 1, 28, 28)
        return out
