import torch

from src.legacy.scvae import supervised_contrastive_loss


torch.manual_seed(42)

# --------------------------------------------------
# Test 1: Random embeddings
# --------------------------------------------------

# The implementation uses two views per sample: [batch, views, features].
features_random = torch.randn(12, 2, 32)

labels = torch.tensor([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])

loss_random = supervised_contrastive_loss(
    features_random,
    labels
)

print("=" * 60)
print("SUPERVISED CONTRASTIVE LOSS TEST")
print("=" * 60)

print(
    "Random embeddings loss:",
    round(loss_random.item(), 4)
)


# --------------------------------------------------
# Test 2: Construct embeddings where same-class
# samples are deliberately very similar.
# --------------------------------------------------

class_centers = torch.randn(3, 32)

features_good = []

for label in labels:

    embedding = (
        class_centers[label]
        + 0.01 * torch.randn(32)
    )

    features_good.append(
        embedding
    )

features_good = torch.stack(features_good)

# Two lightly perturbed views of each class-clustered embedding.
features_good = torch.stack(
    [
        features_good + 0.01 * torch.randn_like(features_good),
        features_good + 0.01 * torch.randn_like(features_good),
    ],
    dim=1,
)

loss_good = supervised_contrastive_loss(
    features_good,
    labels
)

print(
    "Same-class clustered loss:",
    round(loss_good.item(), 4)
)


# --------------------------------------------------
# Test 3: Destroy the class structure
# --------------------------------------------------

permutation = torch.randperm(len(features_good))

features_bad = features_good[
    permutation
]

loss_bad = supervised_contrastive_loss(
    features_bad,
    labels
)

print(
    "Mismatched embeddings loss:",
    round(loss_bad.item(), 4)
)


# --------------------------------------------------
# Basic sanity checks
# --------------------------------------------------

assert torch.isfinite(loss_random)
assert torch.isfinite(loss_good)
assert torch.isfinite(loss_bad)

assert loss_random.item() >= 0
assert loss_good.item() >= 0
assert loss_bad.item() >= 0

print("\nAll basic checks passed.")
