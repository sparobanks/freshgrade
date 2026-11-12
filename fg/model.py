"""The grading model: a small convolutional network, trained on synthetic data.

The network is deliberately small. The images are simple and plentiful, so a
large pre-trained backbone is not needed and would be harder to explain. Three
convolution blocks shrink the 64x64 image to a small feature map, which a
couple of dense layers turn into the three grade scores.

Two things beyond a plain classifier:

* the last convolution layer is kept reachable, so Grad-CAM (in explain.py) can
  show which part of the fruit drove the grade.
* after training, a single temperature is fitted on held-out data so the
  confidence numbers mean what they say (a fruit graded Reject at 0.9 is wrong
  about one time in ten, not far more or less often).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 3


class GradeNet(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 16
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 8
        )
        self.last_conv = self.features[-3]     # the Conv2d in the last block
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(48, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
        self.temperature = 1.0

    def forward(self, x):
        return self.head(self.features(x))

    def logits(self, x):
        return self.forward(x)

    @torch.no_grad()
    def predict_proba(self, x_np: np.ndarray) -> np.ndarray:
        self.eval()
        x = torch.as_tensor(x_np, dtype=torch.float32)
        z = self.forward(x) / self.temperature
        return F.softmax(z, dim=1).cpu().numpy()


def train_model(Xtr, ytr, Xval, yval, epochs: int = 12, seed: int = 0,
                class_weight: np.ndarray | None = None) -> GradeNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GradeNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)
    n = len(Xtr_t)
    bs = 128
    best_state, best_val = None, 1e9
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        # early-stopping on validation loss
        model.eval()
        with torch.no_grad():
            vz = model(torch.as_tensor(Xval, dtype=torch.float32))
            vl = F.cross_entropy(vz, torch.as_tensor(yval, dtype=torch.long)).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    fit_temperature(model, Xval, yval)
    return model


def fit_temperature(model: GradeNet, Xval, yval) -> float:
    """Fit one temperature on the validation set (Guo et al., 2017)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(Xval, dtype=torch.float32))
    y = torch.as_tensor(yval, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=60)

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), y)
        loss.backward()
        return loss
    opt.step(step)
    model.temperature = float(T.detach().clamp(min=0.05).item())
    return model.temperature


def save(model: GradeNet, path):
    torch.save({"state": model.state_dict(), "temperature": model.temperature}, path)


def load(path) -> GradeNet:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = GradeNet()
    model.load_state_dict(blob["state"])
    model.temperature = blob.get("temperature", 1.0)
    model.eval()
    return model
