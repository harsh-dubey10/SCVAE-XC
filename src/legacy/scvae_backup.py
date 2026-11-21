import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Channel Attention / SE Block
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Input:
        [B, C, T]

    C = number of EEG channels
    T = temporal feature dimension
    """

    def __init__(self, channels, reduction=16):

        super().__init__()

        hidden = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Global average pooling over temporal dimension
        squeeze = x.mean(dim=2)

        # Channel weights
        weights = self.fc(squeeze)

        # [B, C] -> [B, C, 1]
        weights = weights.unsqueeze(2)

        return x * weights


# ============================================================
# SCVAE Encoder
# ============================================================

class SCVAEEncoder(nn.Module):
    """
    Encoder for DE-LDS EEG features.

    Input:
        [B, 62, 5]

    Output:
        mu      [B, latent_dim]
        logvar  [B, latent_dim]
    """

    def __init__(
        self,
        channels=62,
        feature_dim=5,
        latent_dim=32
    ):

        super().__init__()

        self.channels = channels
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(

            nn.Conv1d(
                channels,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            SEBlock(128),

            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.flatten_dim = 128 * feature_dim

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(
            64,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            64,
            latent_dim
        )

    def forward(self, x):

        # x: [B, 62, 5]

        h = self.conv(x)

        h = h.reshape(
            h.size(0),
            -1
        )

        h = self.fc(h)

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar


# ============================================================
# SCVAE Decoder
# ============================================================

class SCVAEDecoder(nn.Module):
    """
    Decoder.

    Latent:
        [B, latent_dim]

    Output:
        [B, 62, 5]
    """

    def __init__(
        self,
        channels=62,
        feature_dim=5,
        latent_dim=32
    ):

        super().__init__()

        output_dim = channels * feature_dim

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                output_dim
            )
        )

        self.channels = channels
        self.feature_dim = feature_dim

    def forward(self, z):

        x = self.decoder(z)

        x = x.reshape(
            x.size(0),
            self.channels,
            self.feature_dim
        )

        return x


# ============================================================
# SCVAE
# ============================================================

class SCVAE(nn.Module):
    """
    Supervised Contrastive Variational Autoencoder.

    The model provides:
        - probabilistic latent representation
        - reconstruction
        - KL regularization
        - classification
        - latent representation for MMD
    """

    def __init__(
        self,
        channels=62,
        feature_dim=5,
        latent_dim=32,
        num_classes=3
    ):

        super().__init__()

        self.encoder = SCVAEEncoder(
            channels=channels,
            feature_dim=feature_dim,
            latent_dim=latent_dim
        )

        self.decoder = SCVAEDecoder(
            channels=channels,
            feature_dim=feature_dim,
            latent_dim=latent_dim
        )

        self.classifier = nn.Linear(
            latent_dim,
            num_classes
        )

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return mu + eps * std

    def encode(self, x):

        return self.encoder(x)

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        logits = self.classifier(mu)

        return (
            reconstruction,
            mu,
            logvar,
            z,
            logits
        )


# ============================================================
# Reconstruction Loss
# ============================================================

def reconstruction_loss(
    reconstruction,
    x
):

    return F.mse_loss(
        reconstruction,
        x,
        reduction="mean"
    )


# ============================================================
# KL Divergence
# ============================================================

def kl_loss(
    mu,
    logvar
):

    return -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )


# ============================================================
# Supervised Contrastive Loss
# ============================================================

def supervised_contrastive_loss(
    features,
    labels,
    temperature=0.07
):
    """
    Supervised contrastive loss.

    Samples with the same emotion label are treated
    as positive pairs.

    features:
        [B, latent_dim]

    labels:
        [B]
    """

    device = features.device

    features = F.normalize(
        features,
        dim=1
    )

    similarity = torch.matmul(
        features,
        features.T
    ) / temperature

    batch_size = features.size(0)

    # Remove self-comparisons
    self_mask = torch.eye(
        batch_size,
        device=device,
        dtype=torch.bool
    )

    similarity = similarity.masked_fill(
        self_mask,
        -1e9
    )

    # Same-label mask
    labels = labels.contiguous().view(-1, 1)

    positive_mask = torch.eq(
        labels,
        labels.T
    ).float()

    positive_mask = positive_mask.masked_fill(
        self_mask,
        0.0
    )

    # Log probability
    log_prob = (
        similarity
        - torch.logsumexp(
            similarity,
            dim=1,
            keepdim=True
        )
    )

    positive_count = positive_mask.sum(
        dim=1
    )

    # Avoid division by zero
    valid = positive_count > 0

    mean_log_prob_pos = torch.zeros(
        batch_size,
        device=device
    )

    mean_log_prob_pos[valid] = (
        positive_mask[valid] * log_prob[valid]
    ).sum(dim=1) / positive_count[valid]

    loss = -mean_log_prob_pos[valid].mean()

    return loss


# ============================================================
# MMD Loss
# ============================================================

def gaussian_kernel(
    x,
    y,
    sigma=1.0
):

    x = x.unsqueeze(1)

    y = y.unsqueeze(0)

    distance = (
        (x - y) ** 2
    ).sum(dim=2)

    return torch.exp(
        -distance / (
            2 * sigma ** 2
        )
    )


def mmd_loss(
    source,
    target,
    sigma=1.0
):
    """
    Maximum Mean Discrepancy between
    source and target latent distributions.
    """

    k_ss = gaussian_kernel(
        source,
        source,
        sigma
    )

    k_tt = gaussian_kernel(
        target,
        target,
        sigma
    )

    k_st = gaussian_kernel(
        source,
        target,
        sigma
    )

    return (
        k_ss.mean()
        + k_tt.mean()
        - 2 * k_st.mean()
    )


# ============================================================
# Classification Loss
# ============================================================

def classification_loss(
    logits,
    labels
):

    return F.cross_entropy(
        logits,
        labels
    )


# ============================================================
# Combined SCVAE Loss
# ============================================================

def scvae_loss(
    source_x,
    source_y,
    target_x,
    source_reconstruction,
    target_reconstruction,
    source_mu,
    source_logvar,
    target_mu,
    target_logvar,
    source_logits,
    lambda_recon=1.0,
    lambda_kl=1.0,
    lambda_cls=1.0,
    lambda_supcon=1.0,
    lambda_mmd=1.0
):

    # -------------------------
    # Reconstruction
    # -------------------------

    recon_source = reconstruction_loss(
        source_reconstruction,
        source_x
    )

    recon_target = reconstruction_loss(
        target_reconstruction,
        target_x
    )

    recon = (
        recon_source
        + recon_target
    )

    # -------------------------
    # KL
    # -------------------------

    kl_source = kl_loss(
        source_mu,
        source_logvar
    )

    kl_target = kl_loss(
        target_mu,
        target_logvar
    )

    kl = (
        kl_source
        + kl_target
    )

    # -------------------------
    # Classification
    # -------------------------

    cls = classification_loss(
        source_logits,
        source_y
    )

    # -------------------------
    # Supervised contrastive
    # -------------------------

    supcon = supervised_contrastive_loss(
        source_mu,
        source_y
    )

    # -------------------------
    # MMD
    # -------------------------

    mmd = mmd_loss(
        source_mu,
        target_mu
    )

    # -------------------------
    # Total
    # -------------------------

    total = (
        lambda_recon * recon
        + lambda_kl * kl
        + lambda_cls * cls
        + lambda_supcon * supcon
        + lambda_mmd * mmd
    )

    losses = {
        "total": total,
        "reconstruction": recon,
        "kl": kl,
        "classification": cls,
        "supcon": supcon,
        "mmd": mmd
    }

    return losses
