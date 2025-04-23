import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, image_size: int = 28, latent_dim: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size**2, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size**2),
            nn.Sigmoid(),
        )
        self.image_size = image_size
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        return h.view(-1, 1, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        annealing_factor: float = None,
    ) -> torch.Tensor:
        BCE = nn.functional.binary_cross_entropy(
            x_recon.view(-1, self.image_size**2),
            x.view(-1, self.image_size**2),
            reduction="sum",
        )
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if annealing_factor is not None:
            annealing_factor = min(1, annealing_factor)
            KLD *= annealing_factor

        return BCE + KLD
