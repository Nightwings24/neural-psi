"""
Neural-PSI presentation demo — backend pipeline.

Wraps the real, validated components of the project so the Streamlit app stays
presentation logic only:

    Stage 0  raw SOCOFing fingerprint image (dataset/SOCOFing)
    Stage 1  CNN feature extractor        -> 16-D float embedding   (src/model.py)
    Stage 2  Super-Bit quantiser          -> 128-bit binary code    (src/quantizer.py)
    Stage 3  Fuzzy-Labelled PSI match     -> matched record / reject

Stage 3 can run either through the exact closed-form sub-sampling model used
throughout the evaluation, or through the actual flash-psi Rust binary
(masked-OPRF + garbled circuits + VOLE + Shamir).

Nothing here is demo-only maths: the model checkpoint, the quantiser and the
protocol parameters are the same ones behind the reported results.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from math import comb

import numpy as np
import torch
from PIL import Image

# --- locate the repository and reuse the real implementation modules ---------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
WEEK1 = os.path.join(REPO, "src")
if WEEK1 not in sys.path:
    sys.path.insert(0, WEEK1)

from model import FeatureModel  # noqa: E402
from quantizer import (  # noqa: E402
    NeuralPSIQuantizer,
    QuantizerConfig,
    hamming,
    separation,
)

DATA = os.path.join(WEEK1, "data")
CKPT = os.path.join(DATA, "feature_model_224.pt")
IDS_TEST = os.path.join(DATA, "ids_test_224.npy")

SOCOFING = os.path.join(REPO, "dataset", "SOCOFing")
REAL_DIR = os.path.join(SOCOFING, "Real")
ALTERED_DIR = os.path.join(SOCOFING, "Altered", "Altered-Easy")

FPSI_BIN = os.environ.get("FLASH_PSI_BIN", os.path.join(
    REPO, "crypto", "flash-psi", "target", "release", "fingerprint"))

IMG = 224          # production model resolution
D_BITS = 128       # FLPSI code length (the backend is built for exactly 128)
EMB_DIM = 16       # CNN embedding dimension

ID_RE = re.compile(r"(\d+)__[MF]_(Left|Right)_(\w+?)_finger", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Dataset indexing
# --------------------------------------------------------------------------- #
def identity_of(basename: str):
    """'100__M_Left_index_finger.BMP' -> '100_left_index' (matches ids_test_224)."""
    m = ID_RE.search(basename)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}".lower()


def index_dataset():
    """Map identity -> enrolment (Real) path and probe (Altered-Easy) variants."""
    real, altered = {}, {}
    for fn in sorted(os.listdir(REAL_DIR)):
        if not fn.upper().endswith(".BMP"):
            continue
        ident = identity_of(fn)
        if ident:
            real[ident] = os.path.join(REAL_DIR, fn)
    for fn in sorted(os.listdir(ALTERED_DIR)):
        if not fn.upper().endswith(".BMP"):
            continue
        ident = identity_of(fn)
        if ident is None:
            continue
        stem = os.path.splitext(fn)[0]
        variant = stem.rsplit("_", 1)[-1]          # CR / Obl / Zcut
        altered.setdefault(ident, []).append((variant, os.path.join(ALTERED_DIR, fn)))
    for v in altered.values():
        v.sort()
    return real, altered


def held_out_identities():
    """The 1,200 identities never seen during CNN training."""
    return [str(x) for x in np.load(IDS_TEST, allow_pickle=True)]


VARIANT_LABEL = {
    "CR": "Central rotation",
    "Obl": "Obliteration",
    "Zcut": "Z-cut",
}


# --------------------------------------------------------------------------- #
# Stage 0 / Stage 1 — image loading and CNN embedding
# --------------------------------------------------------------------------- #
def load_gray(path: str, size: int = IMG) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.uint8)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureModel(img=IMG).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    return model, device


def embed_images(model, device, images: np.ndarray, batch: int = 64) -> np.ndarray:
    """images: (N, IMG, IMG) uint8 -> (N, 16) float embeddings."""
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            arr = images[i:i + batch].astype(np.float32) / 255.0
            xb = torch.from_numpy(arr).unsqueeze(1).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, EMB_DIM), dtype=np.float32)


def embed_paths(model, device, paths, batch: int = 64) -> np.ndarray:
    if not paths:
        return np.zeros((0, EMB_DIM), dtype=np.float32)
    imgs = np.stack([load_gray(p) for p in paths])
    return embed_images(model, device, imgs, batch=batch)


# --------------------------------------------------------------------------- #
# Stage 2 — Super-Bit quantiser
# --------------------------------------------------------------------------- #
def fit_quantizer(model, device, calibration_paths, seed: int = 0):
    """
    Fit the public constants (mu, P, tau) on CALIBRATION identities only.

    Calibration uses identities that are not in the held-out test set, so the
    enrolment/probe identities shown in the demo never influence the quantiser.
    """
    emb = embed_paths(model, device, calibration_paths)
    cfg = QuantizerConfig(center=True, whiten=False, superbit=True, balance=True,
                          code_len=D_BITS, in_dim=EMB_DIM, seed=seed)
    return NeuralPSIQuantizer.fit(emb, np.ones(EMB_DIM), cfg), emb


# --------------------------------------------------------------------------- #
# Stage 3 — Fuzzy-Labelled PSI decision
# --------------------------------------------------------------------------- #
def q_clean(h: int, weight: int, d: int = D_BITS) -> float:
    """P[one weight-subset of d positions avoids all h mismatched bits]."""
    h = int(h)
    if d - h < weight:
        return 0.0
    return comb(d - h, weight) / comb(d, weight)


def accept_probability(h: int, weight: int, t: int, T: int) -> float:
    """P[at least t of T sub-samples collide] for a pair at Hamming distance h."""
    p = q_clean(h, weight)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return float(sum(comb(T, j) * p ** j * (1 - p) ** (T - j) for j in range(t, T + 1)))


def bits_to_str(code: np.ndarray) -> str:
    return "".join("1" if int(b) else "0" for b in np.asarray(code).ravel())


@dataclass
class RealPSIResult:
    matched_indices: list
    duration_ms: float


def run_real_flpsi(query_code, db_codes, weight: int, t: int, T: int,
                   timeout: int = 300) -> RealPSIResult:
    """Run the actual flash-psi protocol binary on these codes."""
    if not os.path.exists(FPSI_BIN):
        raise FileNotFoundError(
            f"flash-psi binary not found at {FPSI_BIN}.\n"
            "See crypto/README.md to build it, or set FLASH_PSI_BIN."
        )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        path = fh.name
        fh.write(f"{len(db_codes)} {D_BITS}\n")
        fh.write(bits_to_str(query_code) + "\n")
        for row in db_codes:
            fh.write(bits_to_str(row) + "\n")
    try:
        proc = subprocess.run(
            [FPSI_BIN, path, "-w", str(weight), "-s", str(T), "-t", str(t)],
            capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(path)

    if proc.returncode != 0:
        raise RuntimeError(f"flash-psi failed: {proc.stderr.strip()[:400]}")

    matched, duration = [], float("nan")
    for line in proc.stdout.splitlines():
        if line.startswith("MATCHED_INDICES"):
            body = line.split("[", 1)[1].rsplit("]", 1)[0].strip()
            matched = [int(x) for x in body.split(",")] if body else []
        elif line.startswith("DURATION_MS"):
            duration = float(line.split()[1])
    return RealPSIResult(matched_indices=matched, duration_ms=duration)


# --------------------------------------------------------------------------- #
# Visual helpers
# --------------------------------------------------------------------------- #
def bit_grid_image(code, rows: int = 8, cell: int = 22, diff_mask=None,
                   on=(31, 78, 121), off=(226, 232, 240), changed=(200, 63, 60)):
    """Render a 128-bit code as a rows x (d/rows) grid; optionally flag differing bits."""
    code = np.asarray(code).ravel().astype(int)
    d = len(code)
    cols = d // rows
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    grid[:, :] = (255, 255, 255)
    for k in range(d):
        r, c = divmod(k, cols)
        colour = on if code[k] else off
        if diff_mask is not None and diff_mask[k]:
            colour = changed
        y0, x0 = r * cell, c * cell
        grid[y0 + 1:y0 + cell - 1, x0 + 1:x0 + cell - 1] = colour
    return Image.fromarray(grid)


def fingerprint_image(path: str, size: int = 220) -> Image.Image:
    return Image.open(path).convert("L").resize((size, size), Image.LANCZOS)


__all__ = [
    "REPO", "SOCOFING", "FPSI_BIN", "IMG", "D_BITS", "EMB_DIM", "VARIANT_LABEL",
    "identity_of", "index_dataset", "held_out_identities",
    "load_gray", "load_model", "embed_images", "embed_paths",
    "fit_quantizer", "q_clean", "accept_probability", "bits_to_str",
    "run_real_flpsi", "RealPSIResult", "bit_grid_image", "fingerprint_image",
    "hamming", "separation",
]
