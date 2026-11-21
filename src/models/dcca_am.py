"""DCCA-AM (Liu et al., J. Neural Eng. 2022), ported from the released
CrossCultureCode ``DCCA-AM/dcca_utils.py`` and ``chinese_dcca_attention.py``.

Kept as released: the network, the NumPy CCA gradient (weight -0.1), RMSprop
with the overlapping parameter slices, batches of 30 including the final
partial batch, and skipping a batch that raises. ``train_epoch`` can shuffle
the windows each epoch (``generator``); the released within-person script did
not, but its cross-culture code is unreleased and the unshuffled, clip-ordered
stream underfits the training data. Two further changes: an empty
final batch (when the training size is a multiple of 30) is skipped rather
than fed to the network, where it would give a NaN loss; and the
attention fusion is batched instead of a per-sample loop (same maths; tested
against the loop). ``predict`` skips the CCA step, which does not affect the
logits.
"""

import numpy as np
import torch
import torch.nn as nn

OUTPUT_DIM = 12
LEARNING_RATE = 1e-4
BATCH_SIZE = 30
CCA_GRAD_WEIGHT = -0.1
# The released code skips failing batches silently; a few (a 1-sample final batch) are expected, but
# more than this fraction in one epoch means a systematic error, so the run stops instead.
MAX_FAILED_FRACTION = 0.05


def cca_metric_derivative(H1, H2):
    """Released ``cca_metric_derivative``: (-corr, d corr / d H1, d corr / d H2), rows = samples."""
    r1 = 1e-3
    r2 = 1e-3
    H1 = H1.T
    H2 = H2.T
    o1 = o2 = H1.shape[0]
    m = H1.shape[1]

    H1bar = H1 - H1.mean(axis=1).reshape([-1, 1])
    H2bar = H2 - H2.mean(axis=1).reshape([-1, 1])

    SigmaHat12 = (1.0 / (m - 1)) * np.matmul(H1bar, H2bar.T)
    SigmaHat11 = (1.0 / (m - 1)) * np.matmul(H1bar, H1bar.T) + r1 * np.eye(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.matmul(H2bar, H2bar.T) + r2 * np.eye(o2)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)

    posInd1 = np.where(D1 > 0)[0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = np.where(D2 > 0)[0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    SigmaHat11RootInv = np.matmul(np.matmul(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.matmul(np.matmul(V2, np.diag(D2 ** -0.5)), V2.T)
    Tval = np.matmul(np.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
    tmp = np.matmul(Tval.T, Tval)
    corr = np.sqrt(np.trace(tmp))
    cca_loss = -1 * corr

    U_t, D_t, V_prime_t = np.linalg.svd(Tval)
    Delta12 = SigmaHat11RootInv @ U_t @ V_prime_t @ SigmaHat22RootInv
    Delta11 = SigmaHat11RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat11RootInv
    Delta22 = SigmaHat22RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat22RootInv
    Delta11 = -0.5 * Delta11
    Delta22 = -0.5 * Delta22

    DerivativeH1 = (1.0 / (m - 1)) * (2 * (Delta11 @ H1bar) + Delta12 @ H2bar)
    DerivativeH2 = (1.0 / (m - 1)) * (2 * (Delta22 @ H2bar) + Delta12 @ H1bar)
    return cca_loss, DerivativeH1.T, DerivativeH2.T


class TransformLayers(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        sizes = [input_size] + list(layer_sizes)
        layers = []
        for i in range(len(sizes) - 1):
            if i == len(sizes) - 2:
                layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i + 1])))
            else:
                layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Sigmoid()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionFusion(nn.Module):
    """alpha_1 = exp(x1.w) / (exp(x1.w) + exp(x2.w)); fused = alpha_1 x1 + (1 - alpha_1) x2."""

    def __init__(self, output_dim):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(output_dim, requires_grad=True))

    def forward(self, x1, x2):
        alpha = torch.softmax(torch.stack([x1 @ self.attention_weights, x2 @ self.attention_weights], dim=1), dim=1)
        return alpha[:, :1] * x1 + alpha[:, 1:] * x2, alpha


class DCCA_AM(nn.Module):
    def __init__(self, eeg_dim, eye_dim, layer_sizes, categories):
        super().__init__()
        self.model1 = TransformLayers(eeg_dim, layer_sizes)
        self.model2 = TransformLayers(eye_dim, layer_sizes)
        self.classification = nn.Linear(OUTPUT_DIM, categories)
        self.attention_fusion = AttentionFusion(OUTPUT_DIM)

    def forward(self, x1, x2):
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        cca_loss, partial_h1, partial_h2 = cca_metric_derivative(
            output1.detach().cpu().numpy(), output2.detach().cpu().numpy())
        fused, _ = self.attention_fusion(output1, output2)
        return self.classification(fused), cca_loss, output1, output2, partial_h1, partial_h2

    @torch.no_grad()
    def predict(self, x1, x2):
        fused, _ = self.attention_fusion(self.model1(x1), self.model2(x2))
        return self.classification(fused).argmax(dim=1)


def make_optimizers(model):
    """Released: one RMSprop on all parameters, two on overlapping slices at half the rate."""
    params = list(model.parameters())
    return (torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE),
            torch.optim.RMSprop(params[0:8], lr=LEARNING_RATE / 2),
            torch.optim.RMSprop(params[8:16], lr=LEARNING_RATE / 2))


def train_epoch(model, optimizers, x1, x2, y, batch_size=BATCH_SIZE, generator=None):
    """One epoch: batches of 30, then the remainder (possibly empty). Returns skipped batches.

    With ``generator`` the windows are permuted first (our default for the
    re-run); without it the order is as released (clip by clip, unshuffled).
    """
    if generator is not None:
        order = torch.randperm(len(y), generator=generator).to(y.device)
        x1, x2, y = x1[order], x2[order], y[order]
    classifier_opt, opt1, opt2 = optimizers
    loss_fn = nn.CrossEntropyLoss()
    batches = len(y) // batch_size
    skipped = 0
    first_error = None
    model.train()
    for b in range(batches + 1):
        rows = slice(batch_size * b, None if b == batches else batch_size * (b + 1))
        if len(y[rows]) == 0:  # released code would feed an empty batch (NaN loss); not a failure, just skip it
            continue
        try:
            logits, _, output1, output2, partial_h1, partial_h2 = model(x1[rows], x2[rows])
            loss = loss_fn(logits, y[rows])
            opt1.zero_grad()
            opt2.zero_grad()
            classifier_opt.zero_grad()
            output1.backward(CCA_GRAD_WEIGHT * torch.as_tensor(partial_h1, dtype=torch.float, device=output1.device),
                             retain_graph=True)
            output2.backward(CCA_GRAD_WEIGHT * torch.as_tensor(partial_h2, dtype=torch.float, device=output2.device),
                             retain_graph=True)
            loss.backward()
            opt1.step()
            opt2.step()
            classifier_opt.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                break  # diverged: every further batch would fail; the caller records it
        except Exception as error:  # released code: `except: continue`
            skipped += 1
            first_error = first_error or repr(error)
    if skipped > max(1, MAX_FAILED_FRACTION * (batches + 1)):
        raise RuntimeError(f"{skipped} of {batches + 1} batches failed in one epoch; first error: {first_error}")
    return skipped
