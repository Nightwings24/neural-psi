"""NeuralPSI binariser — 16-D CNN embedding -> 128-bit Hamming code for Fuzzy-Labelled PSI.

This turns the Blind-Touch per-fingerprint embedding ``e in R^16`` into a binary code
``b in {0,1}^128`` whose *Hamming* distance tracks the model's *discriminative* similarity,
so the code can be matched by a fuzzy-labelled PSI backend (Bui-Cong 2025) instead of the
homomorphic-encryption layer of Blind-Touch.

Pipeline (every stage is a FIXED, PUBLIC affine-then-sign map — no learned non-linearity):

    e   = W^T x_hat + bias16                produced upstream (see extract_embedding)
    e0  = e - mu                            centering (public population mean)
    e1  = diag(sqrt(|w1|)) @ e0             metric whitening: fold the head's learned
                                            per-dimension weights w1 so Hamming tracks the
                                            weighted-squared distance the model accepts on,
                                            not raw cosine  (Kulis-Jain learned-metric LSH)
    z   = P @ e1                            Super-Bit LSH: 128x16, orthonormalised in
                                            L=8 blocks of N=16 (variance-reduced SimHash)
    b_k = 1 if z_k > tau_k else 0           per-bit median balancing -> {0,1}^128

For inference we pre-compose the public map offline:

    Pprime = P @ diag(sqrt(|w1|))           (128 x 16)
    z      = Pprime @ (e - mu)              one matmul, then compare to tau

SECURITY NOTE: ``Pprime``, ``mu`` and ``tau`` are PUBLIC. The sign map is NOT one-way:
128 sign measurements over-determine a 16-D vector, so 1-bit compressed sensing recovers
the *direction* of ``e``. Privacy must come from the FLPSI/OPRF layer keeping ``b`` hidden,
never from irreversibility of this map.

Pure NumPy; no TensorFlow needed here (embeddings are produced by the caller). Run
``python3 quantizer.py`` for a synthetic self-test.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

CODE_LEN = 128   # d  — FLPSI code length
IN_DIM = 16      # dimensionality of the CNN embedding e


# --------------------------------------------------------------------------- #
# Public projection matrix
# --------------------------------------------------------------------------- #
def make_superbit_matrix(code_len: int = CODE_LEN, in_dim: int = IN_DIM,
                         seed: int = 0) -> np.ndarray:
    """Super-Bit LSH projection (Ji et al., NIPS 2012).

    Builds ``ceil(code_len / in_dim)`` blocks; within each block ``in_dim`` i.i.d.
    Gaussian vectors are Gram-Schmidt-orthonormalised (via QR). Super-Bit depth
    ``N = in_dim`` is the maximum/optimal depth here (the construction requires
    ``N <= in_dim``). Returns a ``(code_len, in_dim)`` matrix of unit-norm rows.

    With ``N = 1`` blocks this degenerates exactly to plain SimHash; see
    :func:`make_gaussian_matrix` for that baseline.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(code_len / in_dim))
    rows = []
    for _ in range(n_blocks):
        g = rng.standard_normal((in_dim, in_dim))
        q, _ = np.linalg.qr(g)          # columns of q are orthonormal
        rows.append(q.T)                # -> orthonormal rows
    P = np.vstack(rows)[:code_len]
    return P


def make_gaussian_matrix(code_len: int = CODE_LEN, in_dim: int = IN_DIM,
                         seed: int = 0) -> np.ndarray:
    """Plain SimHash / sign-random-projection matrix: i.i.d. Gaussian rows (unit-normed).

    This is the paper's current Section 4.2 choice, kept as the ablation baseline.
    """
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((code_len, in_dim))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


# --------------------------------------------------------------------------- #
# Quantiser
# --------------------------------------------------------------------------- #
@dataclass
class QuantizerConfig:
    """Toggles for the ablation ladder. The full recommended pipeline is all-True."""
    center: bool = True       # subtract the population mean (stage 0)
    whiten: bool = True       # diag(sqrt(|w1|)) metric whitening (stage 1)
    superbit: bool = True     # orthonormalised projection; False -> plain SimHash (stage 2)
    balance: bool = True      # per-bit median thresholds; False -> threshold at 0 (stage 3)
    code_len: int = CODE_LEN
    in_dim: int = IN_DIM
    seed: int = 0


class NeuralPSIQuantizer:
    """Fits the public constants (mu, Pprime, tau) on a calibration set, then transforms.

    All fitted constants are PUBLIC and serialisable. ``transform`` is a single matmul
    followed by a per-bit threshold compare.
    """

    def __init__(self, mu: np.ndarray, Pprime: np.ndarray, tau: np.ndarray,
                 config: QuantizerConfig):
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.Pprime = np.asarray(Pprime, dtype=np.float64)
        self.tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        self.config = config
        assert self.Pprime.shape == (config.code_len, config.in_dim)
        assert self.mu.shape == (config.in_dim,)
        assert self.tau.shape == (config.code_len,)

    # -- fitting ----------------------------------------------------------- #
    @classmethod
    def fit(cls, calib_embeddings: np.ndarray, w1: np.ndarray,
            config: QuantizerConfig | None = None) -> "NeuralPSIQuantizer":
        """Estimate the public constants from a calibration split of embeddings.

        Args:
            calib_embeddings: ``(N, in_dim)`` array of e = W^T x_hat + bias16.
            w1: ``(in_dim,)`` final Dense(1) weights (model.weights[32]); ``|w1|`` is
                used (w1 is typically negative: match -> small diff -> high sigmoid).
            config: ablation toggles; defaults to the full recommended pipeline.
        """
        config = config or QuantizerConfig()
        E = np.asarray(calib_embeddings, dtype=np.float64)
        assert E.ndim == 2 and E.shape[1] == config.in_dim, E.shape
        w1 = np.asarray(w1, dtype=np.float64).reshape(-1)
        assert w1.shape == (config.in_dim,), w1.shape

        mu = E.mean(axis=0) if config.center else np.zeros(config.in_dim)

        if config.superbit:
            P = make_superbit_matrix(config.code_len, config.in_dim, config.seed)
        else:
            P = make_gaussian_matrix(config.code_len, config.in_dim, config.seed)

        if config.whiten:
            scale = np.sqrt(np.abs(w1))             # diag(sqrt(|w1|))
        else:
            scale = np.ones(config.in_dim)
        Pprime = P * scale[np.newaxis, :]           # P @ diag(scale), column scaling

        z = (E - mu) @ Pprime.T                     # (N, code_len)
        tau = np.median(z, axis=0) if config.balance else np.zeros(config.code_len)

        return cls(mu, Pprime, tau, config)

    # -- transform --------------------------------------------------------- #
    def transform(self, e: np.ndarray) -> np.ndarray:
        """Map embedding(s) ``e`` of shape ``(..., in_dim)`` to ``uint8`` codes ``(..., code_len)``."""
        e = np.asarray(e, dtype=np.float64)
        z = (e - self.mu) @ self.Pprime.T
        return (z > self.tau).astype(np.uint8)

    # -- persistence ------------------------------------------------------- #
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "mu.npy"), self.mu)
        np.save(os.path.join(out_dir, "P_public.npy"), self.Pprime)
        np.save(os.path.join(out_dir, "tau.npy"), self.tau)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, in_dir: str) -> "NeuralPSIQuantizer":
        with open(os.path.join(in_dir, "meta.json")) as f:
            config = QuantizerConfig(**json.load(f))
        mu = np.load(os.path.join(in_dir, "mu.npy"))
        Pprime = np.load(os.path.join(in_dir, "P_public.npy"))
        tau = np.load(os.path.join(in_dir, "tau.npy"))
        return cls(mu, Pprime, tau, config)


# --------------------------------------------------------------------------- #
# Hamming helpers
# --------------------------------------------------------------------------- #
def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance between two 0/1 code arrays of equal shape ``(..., d)``."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return np.count_nonzero(a != b, axis=-1)


def separation(genuine_h: np.ndarray, impostor_h: np.ndarray) -> float:
    """d' = (mu_imp - mu_gen) / sqrt((var_gen + var_imp) / 2)  — higher is better."""
    g = np.asarray(genuine_h, dtype=np.float64)
    i = np.asarray(impostor_h, dtype=np.float64)
    denom = np.sqrt((g.var() + i.var()) / 2.0)
    return float((i.mean() - g.mean()) / denom) if denom > 0 else float("inf")


# --------------------------------------------------------------------------- #
# Embedding extraction (used inside the client container; needs the trained model)
# --------------------------------------------------------------------------- #
def extract_embedding(feature_model, W: np.ndarray, images: np.ndarray,
                      bias16: np.ndarray | None = None) -> np.ndarray:
    """Per-fingerprint embedding ``e = W^T x_hat`` (the symmetric quantity to hash).

    ``feature_model`` is the Keras feature extractor; ``W`` = model.weights[30] (25088x16).

    NOTE on bias16 (model.weights[31]): the trained head applies Dense(16) to the
    *difference* x_hat_q - x_hat_db, i.e. it computes ``W^T(x_hat_q - x_hat_db) + bias16``;
    the enroll/auth notebooks reproduce that asymmetrically (only the query carries +bias16).
    FLPSI instead hashes each fingerprint INDEPENDENTLY with the SAME public map, so the
    per-item embedding must be SYMMETRIC: we use ``e = W^T x_hat``. ``bias16`` is a constant
    that is fully absorbed by the centering step (``mu``), so passing it changes nothing after
    fit; it is accepted only for exact parity experiments. Returns ``(N, 16)``.
    """
    feats = feature_model.predict(images, verbose=0)            # (N, 7, 7, 512)
    flat = feats.reshape(feats.shape[0], -1)                    # (N, 25088)
    x_hat = flat / np.linalg.norm(flat, axis=1, keepdims=True)  # UnitNormalization
    e = x_hat @ np.asarray(W)
    if bias16 is not None:
        e = e + np.asarray(bias16).reshape(-1)
    return e


# --------------------------------------------------------------------------- #
# Synthetic self-test (runs on the host, no TF / no model required)
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(12345)
    in_dim, n_ids = IN_DIM, 400
    # Per-identity "true" embedding direction.
    centers = rng.standard_normal((n_ids, in_dim))
    # Model a learned head: half the dimensions are RELIABLE (low intra-identity noise,
    # so they discriminate) and half are NOISY. A trained Dense(1) weights the reliable
    # dims more, so |w1| is large there and small on the noisy dims; w1 is negative
    # (match -> small squared diff -> high sigmoid). Whitening by sqrt(|w1|) should then
    # up-weight the reliable dims and IMPROVE separation.
    sigma = np.where(np.arange(in_dim) < in_dim // 2, 0.08, 0.45)   # per-dim noise
    w1 = -(1.0 / sigma ** 2)                                        # reliable dims -> large |w1|

    def make_view(base):
        return base + sigma * rng.standard_normal(base.shape)

    # Calibration set: one view per identity.
    calib = make_view(centers)

    # Genuine pairs: two independent noisy views of the same identity.
    genuine_a = make_view(centers)
    genuine_b = make_view(centers)
    # Impostor pairs: views of different identities.
    perm = rng.permutation(n_ids)
    impostor_a = make_view(centers)
    impostor_b = make_view(centers[perm])

    ladder = [
        ("plain SimHash (baseline)", QuantizerConfig(center=False, whiten=False, superbit=False, balance=False)),
        ("+ centering",             QuantizerConfig(center=True,  whiten=False, superbit=False, balance=False)),
        ("+ median balance",        QuantizerConfig(center=True,  whiten=False, superbit=False, balance=True)),
        ("+ Super-Bit",             QuantizerConfig(center=True,  whiten=False, superbit=True,  balance=True)),
        ("+ whitening (FULL)",      QuantizerConfig(center=True,  whiten=True,  superbit=True,  balance=True)),
    ]

    print(f"{'stage':<28}{'gen H':>10}{'imp H':>10}{'d-prime':>10}{'bal%':>8}")
    print("-" * 66)
    for name, cfg in ladder:
        q = NeuralPSIQuantizer.fit(calib, w1, cfg)
        ca, cb = q.transform(genuine_a), q.transform(genuine_b)
        ia, ib = q.transform(impostor_a), q.transform(impostor_b)
        gh, ih = hamming(ca, cb), hamming(ia, ib)
        dprime = separation(gh, ih)
        bal = float(np.mean(q.transform(calib).mean(axis=0)))   # mean per-bit P(1)
        print(f"{name:<28}{gh.mean():>10.2f}{ih.mean():>10.2f}{dprime:>10.3f}{bal*100:>7.1f}%")
        assert gh.mean() < ih.mean(), f"{name}: genuine should be closer than impostor"

    # Sanity: full pipeline must separate, and balanced bits ~50%.
    q = NeuralPSIQuantizer.fit(calib, w1, QuantizerConfig())
    bits = q.transform(calib)
    per_bit = bits.mean(axis=0)
    assert np.all((per_bit > 0.3) & (per_bit < 0.7)), "median balancing should keep bits ~50/50"
    # round-trip persistence
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        q.save(d)
        q2 = NeuralPSIQuantizer.load(d)
        assert np.array_equal(q.transform(calib), q2.transform(calib)), "save/load mismatch"
    print("\nself-test OK: ladder separates, bits balanced, save/load round-trips.")


if __name__ == "__main__":
    _self_test()
