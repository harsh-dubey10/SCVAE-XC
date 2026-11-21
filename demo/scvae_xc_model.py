"""SCVAE-XC, inference part only (pure PyTorch, no other dependencies).

Input: (n, 343) per-recording normalised features = 310 EEG differential-entropy features (62 channels x 5 bands)
followed by 33 eye-tracking features. Output: (n, 3) probabilities for (negative, neutral, positive).
Parameter names match the trained SCVAE-XC so its weights load directly (decoder and contrastive heads are not needed
for prediction and are not included).
"""

import torch
from torch import nn

EEG_DIM, EYE_DIM, EYE_HIDDEN, LATENT = 310, 33, 64, 256
CLASSES = ["negative", "neutral", "positive"]


class SEBlock(nn.Module):
    """Channel attention (squeeze-and-excitation)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
                                  nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(self.pool(x).squeeze(-1)).unsqueeze(-1)


class EEGEncoder(nn.Module):
    def __init__(self, channels=62):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 128, kernel_size=1)
        self.attention = SEBlock(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)


class SCVAEXC(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.eye_encoder = nn.Sequential(nn.Linear(EYE_DIM, EYE_HIDDEN), nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(128 * 5 + EYE_HIDDEN, LATENT)
        self.classifier = nn.Sequential(nn.Linear(LATENT, LATENT), nn.BatchNorm1d(LATENT), nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5), nn.Linear(LATENT, 3))

    def forward(self, x):
        enc = self.eeg_encoder
        h = enc.relu(enc.conv1(x[:, :EEG_DIM].reshape(-1, 62, 5)))
        h = h + enc.attention(h)
        h = enc.relu(enc.conv2(h)).flatten(start_dim=1)
        h = torch.cat([h, self.eye_encoder(x[:, EEG_DIM:EEG_DIM + EYE_DIM])], dim=1)
        return torch.softmax(self.classifier(self.fc_mu(h)), dim=1)


def load(weights_path, device="cpu"):
    model = SCVAEXC()
    state = torch.load(weights_path, map_location=device)
    missing, _ = model.load_state_dict(state, strict=False)
    assert not missing, f"weights are missing parameters: {missing}"
    return model.to(device).eval()
