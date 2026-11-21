import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Channel Attention / Squeeze-and-Excitation
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Input:
        x: [B, C, T]

    Output:
        x * channel_weights
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
        # Global average pooling over feature/time dimension
        squeeze = x.mean(dim=2)              # [B, C]

        weights = self.fc(squeeze)           # [B, C]
        weights = weights.unsqueeze(2)       # [B, C, 1]

        return x * weights


# ============================================================
# Encoder
# ============================================================

class SCVAEEncoder(nn.Module):
    """
    EEG -> latent distribution.

    Input:
        [B, 62, 5]

    Output:
        mu:     [B, latent_dim]
        logvar: [B, latent_dim]
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

        flatten_dim = 128 * feature_dim

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 64),
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

        x = self.conv(x)

        x = x.reshape(
            x.size(0),
            -1
        )

        x = self.fc(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


# ============================================================
# Decoder
# ============================================================

class SCVAEDecoder(nn.Module):
    """
    Latent representation -> reconstructed EEG feature tensor.

    Input:
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

        self.channels = channels
        self.feature_dim = feature_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(
                128,
                channels * feature_dim
            )
        )

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

    Main pipeline:

        EEG features
             |
             v
        CNN + SE attention
             |
             v
        latent distribution
          /         \
        mu         logvar
        |             |
        |             +----> KL
        |
        +----> classifier
        |
        +----> projection heads
        |
        +----> decoder <---- z
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

        # Emotion classifier operates on deterministic μ
        self.classifier = nn.Linear(
            latent_dim,
            num_classes
        )

        # ----------------------------------------------------
        # Two learned views for supervised contrastive learning
        # ----------------------------------------------------

        self.feature_extractor1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.feature_extractor2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    # --------------------------------------------------------
    # Reparameterization trick
    # --------------------------------------------------------

    @staticmethod
    def reparameterize(mu, logvar):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return mu + eps * std

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    def encode(self, x):

        return self.encoder(x)

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    def decode(self, z):

        return self.decoder(z)

    # --------------------------------------------------------
    # Contrastive views
    # --------------------------------------------------------

    def get_contrastive_views(self, mu):

        view1 = self.feature_extractor1(mu)
        view2 = self.feature_extractor2(mu)

        # [B, 2, latent_dim]
        features = torch.stack(
            [view1, view2],
            dim=1
        )

        return features

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(z)

        # Classifier uses μ rather than sampled z
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
    target
):
    """
    L1 reconstruction loss.

    This matches the reconstruction criterion used
    in the original implementation.
    """

    return F.l1_loss(
        reconstruction,
        target
    )


# ============================================================
# KL Divergence
# ============================================================

def kl_loss(
    mu,
    logvar
):
    """
    KL divergence between:

        q(z|x) = N(mu, sigma)
        p(z)   = N(0, I)

    Returns mean KL per batch.
    """

    kl = -0.5 * (
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    kl = kl.sum(dim=1)

    return kl.mean()


# ============================================================
# Supervised Contrastive Loss
# ============================================================

def supervised_contrastive_loss(
    features,
    labels,
    temperature=0.07
):
    """
    Multi-view supervised contrastive loss.

    Parameters
    ----------
    features : torch.Tensor
        Shape [B, V, D]

        B = batch size
        V = number of views
        D = representation dimension

    labels : torch.Tensor
        Shape [B]

    temperature : float
        Contrastive temperature.

    Important:
        Samples with the same emotion label are positives.

        Different emotion labels are negatives.

        The two learned views of the same sample are
        also treated as positives because they have
        identical labels.
    """

    if features.dim() != 3:
        raise ValueError(
            "features must have shape [B, V, D]"
        )

    if labels.dim() != 1:
        labels = labels.view(-1)

    device = features.device

    batch_size = features.shape[0]
    n_views = features.shape[1]

    if labels.shape[0] != batch_size:
        raise ValueError(
            "Number of labels must match batch size"
        )

    # --------------------------------------------------------
    # Normalize representations
    # --------------------------------------------------------

    features = F.normalize(
        features,
        dim=-1
    )

    # --------------------------------------------------------
    # Flatten views
    #
    # [B, V, D]
    #      ->
    # [B*V, D]
    # --------------------------------------------------------

    contrast_features = torch.cat(
        torch.unbind(features, dim=1),
        dim=0
    )

    # Use every view as an anchor
    anchor_feature = contrast_features

    anchor_count = n_views
    contrast_count = n_views

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    anchor_dot_contrast = torch.div(
        torch.matmul(
            anchor_feature,
            contrast_features.T
        ),
        temperature
    )

    # --------------------------------------------------------
    # Numerical stabilization
    # --------------------------------------------------------

    logits_max, _ = torch.max(
        anchor_dot_contrast,
        dim=1,
        keepdim=True
    )

    logits = (
        anchor_dot_contrast
        - logits_max.detach()
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = labels.contiguous().view(
        -1, 1
    )

    # [B, B]
    label_mask = torch.eq(
        labels,
        labels.T
    ).float().to(device)

    # Repeat for multiple views
    mask = label_mask.repeat(
        anchor_count,
        contrast_count
    )

    # --------------------------------------------------------
    # Remove self-comparisons
    # --------------------------------------------------------

    logits_mask = torch.ones_like(mask)

    logits_mask.fill_diagonal_(0)

    mask = mask * logits_mask

    # --------------------------------------------------------
    # Log probability
    # --------------------------------------------------------

    exp_logits = (
        torch.exp(logits)
        * logits_mask
    )

    log_prob = (
        logits
        - torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True
            ) + 1e-12
        )
    )

    # --------------------------------------------------------
    # Mean over positive samples
    # --------------------------------------------------------

    positive_count = mask.sum(
        dim=1
    )

    # Some batches could theoretically contain a class
    # represented by only one sample. Avoid division by zero.
    positive_count = torch.clamp(
        positive_count,
        min=1.0
    )

    mean_log_prob_pos = (
        mask * log_prob
    ).sum(dim=1) / positive_count

    loss = -mean_log_prob_pos.mean()

    return loss


# ============================================================
# Gaussian Kernel
# ============================================================

def gaussian_kernel(
    x,
    y,
    sigma=1.0
):
    """
    Gaussian/RBF kernel matrix.
    """

    xx = x.unsqueeze(1)
    yy = y.unsqueeze(0)

    distance_sq = (
        (xx - yy) ** 2
    ).sum(dim=2)

    kernel = torch.exp(
        -distance_sq
        / (2 * sigma ** 2)
    )

    return kernel


# ============================================================
# MMD
# ============================================================

def mmd_loss(
    source,
    target,
    sigma=1.0
):
    """
    Biased empirical Gaussian-kernel MMD.

    Measures discrepancy between the source and target
    latent distributions.
    """

    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            "source and target must have shape [B, D]"
        )

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
        - 2.0 * k_st.mean()
    )


# ============================================================
# Classification Loss
# ============================================================

def classification_loss(
    logits,
    labels
):
    """
    Cross-entropy emotion classification loss.
    """

    return F.cross_entropy(
        logits,
        labels
    )


# ============================================================
# Complete SCVAE Loss
# ============================================================

def scvae_loss(
    reconstruction,
    target,
    mu,
    logvar,
    logits,
    labels,
    target_mu=None,
    contrastive_features=None
):
    """
    Complete SCVAE objective.

    Original weighting:

        0.001  * reconstruction
        0.0012 * KL
        1.0    * classification
        0.052  * supervised contrastive
        0.1    * MMD
    """

    # --------------------------------------------------------
    # Individual losses
    # --------------------------------------------------------

    loss_recon = reconstruction_loss(
        reconstruction,
        target
    )

    loss_kl = kl_loss(
        mu,
        logvar
    )

    loss_class = classification_loss(
        logits,
        labels
    )

    # --------------------------------------------------------
    # SupCon
    # --------------------------------------------------------

    if contrastive_features is None:

        # This is intentionally an error rather than silently
        # falling back to single-view contrastive learning.
        raise ValueError(
            "contrastive_features must be supplied "
            "with shape [B, 2, latent_dim]"
        )

    loss_supcon = supervised_contrastive_loss(
        contrastive_features,
        labels
    )

    # --------------------------------------------------------
    # MMD
    # --------------------------------------------------------

    if target_mu is None:
        loss_mmd = torch.zeros(
            1,
            device=mu.device,
            dtype=mu.dtype
        ).squeeze()
    else:
        loss_mmd = mmd_loss(
            mu,
            target_mu
        )

    # --------------------------------------------------------
    # Weighted total
    # --------------------------------------------------------

    total = (
        0.001 * loss_recon
        + 0.0012 * loss_kl
        + 1.0 * loss_class
        + 0.052 * loss_supcon
        + 0.1 * loss_mmd
    )

    return {
        "total": total,
        "recon": loss_recon,
        "kl": loss_kl,
        "classification": loss_class,
        "supcon": loss_supcon,
        "mmd": loss_mmd
    }