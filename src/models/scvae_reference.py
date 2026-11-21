"""Strict reproduction components for the released SCVAE-Net architecture.

The architecture and loss weights mirror the authors' released cross-subject
code.  Training and evaluation policy belongs in the experiment runner: target
labels must never influence checkpoint or epoch selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceSEBlock(nn.Module):
    """Channel attention used by the released SCVAE-Net code."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.pool(x).squeeze(-1)
        weights = self.gate(weights).unsqueeze(-1)
        return x * weights


class ReferenceEncoder(nn.Module):
    """Released-code VAE encoder: Conv1d(1) -> residual SE -> Conv1d(1)."""

    def __init__(self, channels=62, feature_dim=5, latent_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(channels, 128, kernel_size=1)
        self.attention = ReferenceSEBlock(128, reduction=8)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc_mu = nn.Linear(128 * feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * feature_dim, latent_dim)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = h + self.attention(h)
        h = self.relu(self.conv2(h))
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class ReferenceDecoder(nn.Module):
    def __init__(self, channels=62, feature_dim=5, latent_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(latent_dim, 128 * feature_dim)
        self.conv = nn.Conv1d(128, channels, kernel_size=1)

    def forward(self, z):
        h = self.fc(z).reshape(z.shape[0], 128, self.feature_dim)
        return self.conv(h)


class MultiKernelMMD(nn.Module):
    """The released code's five-bandwidth Gaussian MMD, without `.data`."""

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def kernel_matrix(self, source, target):
        """Return the released implementation's multi-bandwidth kernel matrix."""
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source and target must have shape [batch, features]")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source and target must have the same feature dimension")

        total = torch.cat([source, target], dim=0)
        distances = torch.cdist(total, total, p=2).pow(2)
        n = total.shape[0]
        bandwidth = distances.detach().sum() / (n * n - n)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        kernels = sum(
            torch.exp(-distances / (bandwidth * (self.kernel_mul ** index)))
            for index in range(self.kernel_num)
        )
        return kernels

    def forward(self, source, target):
        kernels = self.kernel_matrix(source, target)

        batch_size = source.shape[0]
        xx = kernels[:batch_size, :batch_size].mean()
        yy = kernels[batch_size:, batch_size:].mean()
        xy = kernels[:batch_size, batch_size:].mean()
        yx = kernels[batch_size:, :batch_size].mean()
        return xx + yy - xy - yx


class TargetUncertaintyWeights(nn.Module):
    """Create detached target weights for uncertainty-aware MMD.

    Entropy measures classifier uncertainty. Posterior uncertainty is the mean
    latent variance, mapped into ``[0, 1)`` relative to the current target
    batch. The returned weights are intentionally detached: the encoder and
    classifier must improve alignment, not minimize MMD by changing weights.
    """

    MODES = {
        "entropy_weighted_mmd",
        "posterior_weighted_mmd",
        "combined_uncertainty_mmd",
    }

    def __init__(self, mode, temperature=0.25, uniform_mix=0.2, entropy_mix=0.5):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported uncertainty weighting mode: {mode}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= uniform_mix <= 1:
            raise ValueError("uniform_mix must be in [0, 1]")
        if not 0 <= entropy_mix <= 1:
            raise ValueError("entropy_mix must be in [0, 1]")
        self.mode = mode
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.entropy_mix = entropy_mix

    def forward(self, logits, logvar):
        if logits.ndim != 2 or logvar.ndim != 2:
            raise ValueError("logits and logvar must have shape [batch, features]")
        if logits.shape[0] != logvar.shape[0] or logits.shape[0] < 2:
            raise ValueError("logits and logvar must share a batch dimension of at least 2")

        with torch.no_grad():
            probabilities = torch.softmax(logits.detach(), dim=1).clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=1)
            entropy = entropy / log_num_classes(logits.shape[1], logits)

            mean_variance = logvar.detach().exp().mean(dim=1)
            relative_variance = mean_variance / mean_variance.mean().clamp_min(1e-12)
            posterior_uncertainty = 1.0 - torch.exp(-relative_variance)

            if self.mode == "entropy_weighted_mmd":
                uncertainty = entropy
            elif self.mode == "posterior_weighted_mmd":
                uncertainty = posterior_uncertainty
            else:
                uncertainty = (
                    self.entropy_mix * entropy
                    + (1.0 - self.entropy_mix) * posterior_uncertainty
                )

            focused_weights = torch.softmax(-uncertainty / self.temperature, dim=0)
            uniform_weights = torch.full_like(focused_weights, 1.0 / len(focused_weights))
            weights = self.uniform_mix * uniform_weights + (1.0 - self.uniform_mix) * focused_weights
            diagnostics = {
                "target_entropy": entropy.mean(),
                "target_posterior_uncertainty": posterior_uncertainty.mean(),
                "target_combined_uncertainty": uncertainty.mean(),
                "target_weight_min": weights.min(),
                "target_weight_max": weights.max(),
                "target_weight_effective_samples": 1.0 / weights.square().sum(),
            }
        return weights.detach(), diagnostics


def log_num_classes(num_classes, reference):
    """A device-safe scalar for normalized categorical entropy."""
    return torch.log(torch.tensor(float(num_classes), device=reference.device, dtype=reference.dtype))


class UncertaintyWeightedMMD(MultiKernelMMD):
    """Multi-kernel MMD with uniform source and detached weighted target terms."""

    def forward(self, source, target, target_weights):
        kernels = self.kernel_matrix(source, target)
        source_count = source.shape[0]
        target_count = target.shape[0]
        if target_weights.ndim != 1 or target_weights.shape[0] != target_count:
            raise ValueError("target_weights must have shape [target batch]")
        if not torch.isfinite(target_weights).all() or (target_weights < 0).any():
            raise ValueError("target_weights must be finite and non-negative")

        target_weights = target_weights.detach().to(device=target.device, dtype=target.dtype)
        target_weights = target_weights / target_weights.sum().clamp_min(1e-12)
        source_weights = torch.full(
            (source_count,), 1.0 / source_count, device=source.device, dtype=source.dtype
        )

        xx_kernel = kernels[:source_count, :source_count]
        yy_kernel = kernels[source_count:, source_count:]
        xy_kernel = kernels[:source_count, source_count:]
        xx = source_weights @ xx_kernel @ source_weights
        yy = target_weights @ yy_kernel @ target_weights
        xy = source_weights @ xy_kernel @ target_weights
        return xx + yy - 2.0 * xy


class ReferenceSupConLoss(nn.Module):
    """Two-view supervised contrastive loss from the released implementation."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        if features.ndim < 3:
            raise ValueError("features must have shape [batch, views, features]")
        if features.ndim > 3:
            features = features.flatten(start_dim=2)
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape [batch]")

        batch_size, view_count = features.shape[:2]
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = contrast @ contrast.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        label_mask = labels.view(-1, 1).eq(labels.view(1, -1)).float()
        mask = label_mask.repeat(view_count, view_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        log_prob = logits - torch.log(
            (torch.exp(logits) * logits_mask).sum(dim=1, keepdim=True)
        )
        positive_count = mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_positive = (mask * log_prob).sum(dim=1) / positive_count
        return -(self.temperature / self.base_temperature) * mean_log_prob_positive.mean()


class ReferenceSCVAE(nn.Module):
    """Architecture used for the strict, session-wise SCVAE-Net reproduction."""

    def __init__(self, channels=62, feature_dim=5, latent_dim=256, num_classes=3):
        super().__init__()
        self.encoder = ReferenceEncoder(channels, feature_dim, latent_dim)
        self.decoder = ReferenceDecoder(channels, feature_dim, latent_dim)
        self.view_one = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(inplace=True))
        self.view_two = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(latent_dim, num_classes),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        logits = self.classifier(mu)
        return reconstruction, mu, logvar, logits

    def contrastive_views(self, mu):
        return torch.stack([self.view_one(mu), self.view_two(mu)], dim=1)


def reconstruction_loss(reconstruction, target):
    return F.l1_loss(reconstruction, target)


def kl_loss(mu, logvar):
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))


@torch.no_grad()
def recalibrate_batchnorm(model, x, device, batch_size=512):
    """Re-estimate BatchNorm running statistics from ``x`` without labels.

    Training forwards source and target batches separately in train mode, so
    the saved running statistics are an exponential average dominated by the
    most recent (unshuffled) target batches. Resetting them and taking a
    cumulative average over one fixed domain removes that dependence. Only
    BatchNorm buffers change; no parameters are updated.
    """
    batchnorm_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [layer.momentum for layer in batchnorm_layers]
    for layer in batchnorm_layers:
        layer.reset_running_stats()
        layer.momentum = None
    model.train()
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        if len(batch) > 1:  # train-mode BatchNorm rejects single-sample batches
            model(batch.to(device))
    for layer, momentum in zip(batchnorm_layers, momenta):
        layer.momentum = momentum
    model.eval()
    return model


def class_conditional_mmd(mmd, source_mu, source_y, target_mu, target_logits, threshold=0.9, num_classes=3):
    """Align each emotion separately instead of the whole distributions.

    Target windows get a pseudo-label from the classifier only when it is
    confident (max softmax >= ``threshold``); target labels are never used.
    The loss averages ``mmd`` between source and confident target latents of
    the same class, so emotions cannot be matched to the wrong emotion the way
    class-blind MMD can across cultures. Pseudo-labels are detached.
    Returns the loss and the fraction of confident target windows.
    """
    with torch.no_grad():
        confidence, pseudo = F.softmax(target_logits, dim=1).max(dim=1)
        confident = confidence >= threshold
    losses = []
    for c in range(num_classes):
        s = source_mu[source_y == c]
        t = target_mu[confident & (pseudo == c)]
        if len(s) >= 2 and len(t) >= 2:
            losses.append(mmd(s, t))
    loss = torch.stack(losses).mean() if losses else source_mu.sum() * 0.0
    return loss, confident.float().mean()
