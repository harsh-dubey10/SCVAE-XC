"""Checks for the DCCA-AM port.

    python -m src.models.test_dcca_am
"""

import numpy as np
import torch

from src.models.dcca_am import OUTPUT_DIM, DCCA_AM, cca_metric_derivative, make_optimizers, train_epoch


def attention_loop(x1, x2, w):
    """The released per-sample loop (dcca_utils.AttentionFusion.forward)."""
    fused = torch.empty_like(x1)
    for i in range(len(x1)):
        a = torch.exp(torch.dot(x1[i], w))
        b = torch.exp(torch.dot(x2[i], w))
        alpha_1 = a / (a + b)
        fused[i] = alpha_1 * x1[i] + (1 - alpha_1) * x2[i]
    return fused


def sum_singular_values(H1, H2, r=1e-3):
    """sum of singular values of S11^-1/2 S12 S22^-1/2, the quantity the derivative is for."""
    m = H1.shape[0]
    a, b = H1 - H1.mean(0), H2 - H2.mean(0)
    s12 = a.T @ b / (m - 1)
    s11 = a.T @ a / (m - 1) + r * np.eye(a.shape[1])
    s22 = b.T @ b / (m - 1) + r * np.eye(b.shape[1])

    def root_inv(s):
        d, v = np.linalg.eigh(s)
        return v @ np.diag(d ** -0.5) @ v.T

    return np.linalg.svd(root_inv(s11) @ s12 @ root_inv(s22), compute_uv=False).sum()


torch.manual_seed(0)
rng = np.random.default_rng(0)

# Batched attention equals the released loop.
model = DCCA_AM(310, 33, [150, 35, OUTPUT_DIM], 3)
x1, x2 = torch.randn(40, OUTPUT_DIM), torch.randn(40, OUTPUT_DIM)
fused, _ = model.attention_fusion(x1, x2)
assert torch.allclose(fused, attention_loop(x1, x2, model.attention_fusion.attention_weights), atol=1e-6)

# The CCA derivative is the gradient of the sum of canonical correlations.
H1, H2 = rng.normal(size=(50, 3)), rng.normal(size=(50, 3)) + 0.5 * rng.normal(size=(50, 3))
_, d1, _ = cca_metric_derivative(H1, H2)
E = rng.normal(size=H1.shape)
eps = 1e-6
numeric = (sum_singular_values(H1 + eps * E, H2) - sum_singular_values(H1 - eps * E, H2)) / (2 * eps)
assert abs(numeric - (d1 * E).sum()) < 1e-4 * max(1.0, abs(numeric)), (numeric, (d1 * E).sum())

# One epoch on random data runs, changes the weights, and skips the 1-sample last batch.
x1, x2 = torch.randn(61, 310), torch.randn(61, 33)  # 61 = 2 * 30 + 1: final batch of one
y = torch.randint(0, 3, (61,))
before = [p.detach().clone() for p in model.parameters()]
skipped = train_epoch(model, make_optimizers(model), x1, x2, y)
assert skipped == 1, skipped
assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
assert model.predict(x1, x2).shape == (61,)

# The optimiser parameter slices match the released code (model1 + first layer of model2, then the rest).
params = list(model.parameters())
assert len(params) == 15 and params[0:8][-1] is model.model2.layers[0][0].bias
# Shuffled epoch = released epoch on the data permuted with the same generator.
torch.manual_seed(1)
a = DCCA_AM(310, 33, [150, 35, OUTPUT_DIM], 3)
b = DCCA_AM(310, 33, [150, 35, OUTPUT_DIM], 3)
b.load_state_dict(a.state_dict())
order = torch.randperm(61, generator=torch.Generator().manual_seed(5))
train_epoch(a, make_optimizers(a), x1, x2, y, generator=torch.Generator().manual_seed(5))
train_epoch(b, make_optimizers(b), x1[order], x2[order], y[order])
assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
print("dcca_am checks passed")
