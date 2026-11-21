"""Fast shape and loss smoke tests for the strict SCVAE reference components."""

import torch

from src.models.scvae_reference import (
    MultiKernelMMD,
    ReferenceSCVAE,
    ReferenceSupConLoss,
    TargetUncertaintyWeights,
    UncertaintyWeightedMMD,
    kl_loss,
    reconstruction_loss,
)


torch.manual_seed(12)
source = torch.randn(8, 62, 5)
target = torch.randn(8, 62, 5)
labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

model = ReferenceSCVAE()
source_reconstruction, source_mu, source_logvar, logits = model(source)
target_reconstruction, target_mu, target_logvar, _ = model(target)
views = model.contrastive_views(source_mu)

assert source_reconstruction.shape == source.shape
assert target_reconstruction.shape == target.shape
assert source_mu.shape == (8, 256)
assert views.shape == (8, 2, 128)
assert logits.shape == (8, 3)

losses = [
    reconstruction_loss(source_reconstruction, source),
    kl_loss(source_mu, source_logvar),
    MultiKernelMMD()(source_mu, target_mu),
    # The final target loader batch can be smaller than a source batch.
    MultiKernelMMD()(source_mu, target_mu[:6]),
    ReferenceSupConLoss()(views, labels),
]
assert all(torch.isfinite(loss) for loss in losses)

# Uniform target weights must produce exactly the standard (V-statistic) MMD.
standard_mmd = MultiKernelMMD()(source_mu, target_mu)
uniform_weights = torch.full((len(target_mu),), 1.0 / len(target_mu))
weighted_mmd = UncertaintyWeightedMMD()(source_mu, target_mu, uniform_weights)
assert torch.allclose(standard_mmd, weighted_mmd, atol=1e-6, rtol=1e-5)

for mode in sorted(TargetUncertaintyWeights.MODES):
    logits_for_weights = torch.randn(8, 3, requires_grad=True)
    logvar_for_weights = torch.randn(8, 256, requires_grad=True)
    weights, diagnostics = TargetUncertaintyWeights(mode)(
        logits_for_weights, logvar_for_weights
    )
    assert not weights.requires_grad
    assert weights.shape == (8,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert torch.isfinite(UncertaintyWeightedMMD()(source_mu, target_mu, weights))
print("Reference SCVAE component tests passed.")
