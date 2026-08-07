"""
Week 1 — PyTorch port of the Blind-Touch Siamese CNN.

Faithful to the SOKOTO training notebook's feature extractor:
  5x [Conv(3x3, same) -> BatchNorm -> swish(SiLU) -> MaxPool(2)]
  channels 32 -> 64 -> 128 -> 256 -> 512
  Flatten -> Dense(EMB_DIM=16)            <-- the "FC-16" float vector

The siamese head compares two FC-16 embeddings via |a - b| -> Dense(1) -> sigmoid
and is trained with BCE on genuine(1)/impostor(0) pairs. After training, the
feature extractor alone produces the 16-d float features used downstream
(binarized -> PSI), and for EER measurement.
"""
import torch
import torch.nn as nn

EMB_DIM = 16


class FeatureModel(nn.Module):
    """Maps a (1, H, W) fingerprint to a 16-d float embedding (Blind-Touch FC-16)."""

    def __init__(self, img: int = 96, emb_dim: int = EMB_DIM):
        super().__init__()
        chans = [32, 64, 128, 256, 512]
        blocks = []
        in_c = 1
        for c in chans:
            blocks += [
                nn.Conv2d(in_c, c, kernel_size=3, padding="same"),
                nn.BatchNorm2d(c),
                nn.SiLU(),               # swish
                nn.MaxPool2d(2),
            ]
            in_c = c
        self.features = nn.Sequential(*blocks)
        feat_hw = img // (2 ** len(chans))          # 96 -> 3
        self.flat_dim = chans[-1] * feat_hw * feat_hw
        self.fc = nn.Linear(self.flat_dim, emb_dim)  # FC-16

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)                            # raw float embedding


class SiameseModel(nn.Module):
    """Shared feature extractor + L1-distance verification head (genuine vs impostor)."""

    def __init__(self, img: int = 96, emb_dim: int = EMB_DIM):
        super().__init__()
        self.feature_model = FeatureModel(img, emb_dim)
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x1, x2):
        e1 = self.feature_model(x1)
        e2 = self.feature_model(x2)
        return self.head(torch.abs(e1 - e2)).squeeze(1)   # logit
