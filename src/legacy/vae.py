import torch
from torch import nn


class VAE(nn.Module):
    """
    Variational Autoencoder for SEED DE-LDS features.

    Input:
        [batch, 62, 5]

    Flattened:
        62 * 5 = 310 features

    Latent representation:
        z ~ N(mu, sigma^2)
    """

    def __init__(self, input_dim=310, latent_dim=32):

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # -------------------------
        # Encoder
        # -------------------------

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Mean and log-variance
        self.fc_mu = nn.Linear(64, latent_dim)

        self.fc_logvar = nn.Linear(64, latent_dim)

        # -------------------------
        # Decoder
        # -------------------------

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim)
        )

    def encode(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        # sigma = exp(0.5 * log(sigma^2))
        std = torch.exp(0.5 * logvar)

        # Random epsilon ~ N(0, I)
        eps = torch.randn_like(std)

        # z = mu + sigma * epsilon
        z = mu + eps * std

        return z

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar, z


def vae_loss(
    reconstruction,
    x,
    mu,
    logvar,
    beta=1.0
):
    """
    VAE objective:

        L = reconstruction loss
            + beta * KL divergence

    Reconstruction:
        Mean squared error

    KL:
        KL(q(z|x) || N(0,I))
    """

    reconstruction_loss = nn.functional.mse_loss(
        reconstruction,
        x,
        reduction="mean"
    )

    kl_loss = -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    total_loss = (
        reconstruction_loss
        + beta * kl_loss
    )

    return total_loss, reconstruction_loss, kl_loss
