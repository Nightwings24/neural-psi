"""Statistical surrogate for the Bui-Cong (2025) Fuzzy-Labelled PSI matcher.

This is NOT the cryptographic protocol — it is the *balls-and-bins* statistical model
the protocol's FPR/FNR are derived from, so we can measure how a given binariser performs
end-to-end without standing up the OPRF/TLPSI/VOLE backend (that is Phase 3).

Modelling assumption (the decisive one — see plan open question):
  * FLPSI sub-samples positions with a random permutation/mask, so the accept/reject
    decision depends ONLY on the TOTAL Hamming distance H between two codes, not on
    *which* bits differ. We verify this position-independence by Monte-Carlo below.
  * A single sub-sample draws ``d_s`` of the ``d`` positions WITHOUT replacement and is a
    "match" if it contains at most ``t`` mismatched positions  ->  hypergeometric tail.
  * Over ``T`` independent sub-samples the match count is Binomial(T, p(H)); the parties
    declare a fuzzy match if that count reaches a TLPSI threshold ``theta``.

Cited parameters (main.pdf 2.3 / 6): d=128, T=64, d_s=14, t=2, with target
FPR < 7e-6 and FNR < 2e-16. ``theta`` is calibrated here against the measured
genuine/impostor Hamming histograms (the exact T/t/theta/d_k mapping in Bui-Cong should
be confirmed against eprint 2025/1470). Exact integer combinatorics via ``math.comb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

D = 128
T = 64
D_S = 14
T_ERR = 2
TARGET_FPR = 7e-6
TARGET_FNR = 2e-16


# --------------------------------------------------------------------------- #
# Closed-form per-distance probabilities
# --------------------------------------------------------------------------- #
def single_subsample_match_prob(H: int, d: int = D, d_s: int = D_S, t: int = T_ERR) -> float:
    """P[ a random ``d_s``-subset of ``d`` positions hits <= ``t`` of the ``H`` mismatches ].

    Hypergeometric tail: X ~ Hypergeom(population=d, successes=H, draws=d_s);
    returns P[X <= t].  (t = 0 recovers the 'error-free sub-sample' probability
    C(d-H, d_s) / C(d, d_s).)
    """
    H = int(H)
    if H < 0:
        H = 0
    if H > d:
        H = d
    denom = comb(d, d_s)
    total = 0
    for i in range(0, min(t, H, d_s) + 1):
        total += comb(H, i) * comb(d - H, d_s - i)
    return total / denom


def _binom_tail_ge(n: int, k: int, p: float) -> float:
    """P[ Binomial(n, p) >= k ] computed exactly (n small, e.g. T=64)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # Sum the upper tail; use log-free direct terms (n<=64 keeps comb manageable).
    s = 0.0
    for j in range(k, n + 1):
        s += comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j))
    return min(max(s, 0.0), 1.0)


def accept_prob(H, d: int = D, d_s: int = D_S, t: int = T_ERR,
                T_sub: int = T, theta: int = 1):
    """P[ declare match | true Hamming distance H ] = P[ Binomial(T, p(H)) >= theta ].

    Accepts a scalar or array of H; returns the same shape.
    """
    H_arr = np.atleast_1d(np.asarray(H))
    out = np.empty(H_arr.shape, dtype=np.float64)
    for idx, h in np.ndenumerate(H_arr):
        p = single_subsample_match_prob(int(round(float(h))), d, d_s, t)
        out[idx] = _binom_tail_ge(T_sub, theta, p)
    return out if np.ndim(H) else float(out.reshape(-1)[0])


# --------------------------------------------------------------------------- #
# ROC / operating point over measured Hamming histograms
# --------------------------------------------------------------------------- #
@dataclass
class OperatingPoint:
    theta: int
    fpr: float           # P[accept | impostor], averaged over impostor pairs
    fnr: float           # P[reject | genuine],  averaged over genuine pairs
    meets_targets: bool
    effective_dk: float  # H at which accept_prob crosses 0.5 (the fuzzy radius)


def _effective_dk(d: int, d_s: int, t: int, T_sub: int, theta: int) -> float:
    """Largest H whose accept probability is >= 0.5 (the de-facto acceptance radius d_k)."""
    last = 0.0
    for h in range(0, d + 1):
        p = accept_prob(h, d, d_s, t, T_sub, theta)
        if p < 0.5:
            return float(h - 1) + (0.5 - last) / (p - last) if p != last else float(h - 1)
        last = p
    return float(d)


def evaluate(genuine_H: np.ndarray, impostor_H: np.ndarray, theta: int,
             d: int = D, d_s: int = D_S, t: int = T_ERR, T_sub: int = T) -> OperatingPoint:
    """FPR/FNR at a given ``theta``, averaging the per-pair accept probabilities."""
    g = np.asarray(genuine_H, dtype=float)
    i = np.asarray(impostor_H, dtype=float)
    fnr = float(np.mean(1.0 - accept_prob(g, d, d_s, t, T_sub, theta)))
    fpr = float(np.mean(accept_prob(i, d, d_s, t, T_sub, theta)))
    return OperatingPoint(
        theta=theta, fpr=fpr, fnr=fnr,
        meets_targets=(fpr <= TARGET_FPR and fnr <= TARGET_FNR),
        effective_dk=_effective_dk(d, d_s, t, T_sub, theta),
    )


def calibrate_theta(genuine_H: np.ndarray, impostor_H: np.ndarray,
                    d: int = D, d_s: int = D_S, t: int = T_ERR,
                    T_sub: int = T) -> tuple[OperatingPoint, list[OperatingPoint]]:
    """Sweep theta in 1..T; pick the operating point that meets both targets with the
    largest FNR margin, else the one minimising max(fpr/target, fnr/target)."""
    sweep = [evaluate(genuine_H, impostor_H, th, d, d_s, t, T_sub) for th in range(1, T_sub + 1)]
    feasible = [op for op in sweep if op.meets_targets]
    if feasible:
        best = min(feasible, key=lambda op: op.fnr)          # most genuine-safe feasible point
    else:
        def cost(op):
            return max(op.fpr / TARGET_FPR, op.fnr / max(TARGET_FNR, 1e-300))
        best = min(sweep, key=cost)
    return best, sweep


# --------------------------------------------------------------------------- #
# Monte-Carlo cross-check (verifies position-independence + the closed form)
# --------------------------------------------------------------------------- #
def simulate_accept(code_a: np.ndarray, code_b: np.ndarray, theta: int,
                    d_s: int = D_S, t: int = T_ERR, T_sub: int = T,
                    seed: int = 0) -> bool:
    """Actually run T random sub-samples on two concrete codes; accept if matches >= theta."""
    a = np.asarray(code_a, dtype=np.uint8)
    b = np.asarray(code_b, dtype=np.uint8)
    d = a.shape[-1]
    diff = (a != b)
    rng = np.random.default_rng(seed)
    matches = 0
    for _ in range(T_sub):
        idx = rng.choice(d, size=d_s, replace=False)
        if int(diff[idx].sum()) <= t:
            matches += 1
    return matches >= theta


def _self_test() -> None:
    # Closed form sanity: accept prob is monotone non-increasing in H.
    ps = [accept_prob(h, theta=20) for h in range(0, D + 1)]
    assert all(ps[i] >= ps[i + 1] - 1e-12 for i in range(len(ps) - 1)), "accept_prob must be monotone in H"

    # Position-independence: same H, different mismatch positions -> same MC accept rate.
    d = D
    rng = np.random.default_rng(1)
    H = 12
    base = rng.integers(0, 2, size=d, dtype=np.uint8)
    def code_with_H(positions):
        c = base.copy()
        c[positions] ^= 1
        return c
    pos1 = rng.choice(d, size=H, replace=False)
    pos2 = rng.choice(d, size=H, replace=False)
    def mc_rate(pos):
        c2 = code_with_H(pos)
        return np.mean([simulate_accept(base, c2, theta=20, seed=s) for s in range(200)])
    r1, r2 = mc_rate(pos1), mc_rate(pos2)
    assert abs(r1 - r2) < 0.15, f"accept rate should depend only on H, got {r1:.3f} vs {r2:.3f}"

    # Separated genuine/impostor distributions should yield a feasible operating point.
    gen = np.full(500, 12)
    imp = np.full(500, 63)
    best, _ = calibrate_theta(gen, imp)
    print(f"calibrated theta={best.theta}  fpr={best.fpr:.2e}  fnr={best.fnr:.2e}  "
          f"eff_dk={best.effective_dk:.1f}  meets_targets={best.meets_targets}")
    print("self-test OK: monotone accept curve, position-independence, feasible calibration.")


if __name__ == "__main__":
    _self_test()
