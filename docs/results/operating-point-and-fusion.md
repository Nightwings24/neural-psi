# Tier-2 results — flash-psi operating-point re-tune + multi-finger fusion (no GPU/retrain)

**On the Tier-1 224-model Super-Bit codes (plain, no whitening). Headline: single-finger FAR is
floored ~3% by a thumb-quality tail in SOCOFing; multi-finger fusion crushes it — 2-of-3 gives
TAR 99.9% @ FAR 2.4e-3, 3-of-3 gives FAR 2.4e-5 — through the real flash-psi decision model.**

Run: `python3 08_tier2_fusion.py` (held-out 1,200 fingers; 240k impostor pairs).

## 1. Operating-point re-tune (the old point was wrong)
The 224 model gives much tighter genuine Hamming (mean **8.3/128**, max 33) vs impostor (mean
**64.0/128**), so the 96px-era flash-psi point (`weight=2, t=54`) no longer applies. `t` must stay
small (Shamir cost ~`C(t+1)`), so we fix `t∈{2,3}` and tune `weight` (T=64). ROC at **t=2**:

| weight | TAR | FAR |
|---|---|---|
| 8  | 99.99% | 2.1e-1 |
| 12 | 99.62% | 7.3e-2 |
| **14** | **99.03%** | **4.5e-2** |
| 16 | 98.02% | 2.9e-2 |
| 24 | 89.36% | 6.9e-3 |
| 32 | 76.69% | 2.2e-3 |
| 40 | 63.90% | 8.8e-4 |

**There is no single-finger point with both high TAR and low FAR** — to push FAR below ~1e-3 you must
drop TAR below ~64%. Recommended single-finger base: `weight=14, t=2` (TAR 99%, FAR 4.5%).

## 2. Why single-finger FAR is floored — a SOCOFing thumb-quality tail
**7.12% of impostor pairs (17,082 / 240,000) have Hamming ≤ the genuine max (33)** — a real
distribution overlap. The closest impostor collisions are almost all **`right_thumb` vs `right_thumb`
across different subjects** (e.g. `525_right_thumb` vs `47_right_thumb` at H=2). SOCOFing thumb prints
are low-quality/confusable, so the CNN maps many right-thumbs close together. This is a **data-quality
tail, not a bridge flaw** — and it's exactly what multi-finger fusion neutralises.

## 3. Multi-finger fusion — the decisive lever
Enrol K fingers per person as K independent flash-psi matches; require a **q-of-K quorum**. Independent
fingers ⇒ FAR drops geometrically (one confusable thumb can no longer carry a false accept).

| Rule | Fused TAR | Fused FAR | vs single-finger FAR |
|---|---|---|---|
| single (K=1) | 98.18% | 2.9e-2 | — |
| 2-of-2 (AND) | 96.39% | **8.2e-4** | 35× lower |
| **2-of-3 (majority)** | **99.90%** | **2.4e-3** | 12× lower **and higher TAR** |
| 3-of-3 (all) | 94.64% | **2.4e-5** | **1,200× lower** |

- **2-of-3 is the sweet spot**: it *raises* TAR to 99.9% (tolerates one bad finger) *and* cuts FAR to
  2.4e-3 — strictly better than single-finger on both axes.
- **3-of-3** for high-security: FAR 2.4e-5.
- **Empirical validation** (372 held-out test subjects with ≥2 fingers, real codes through ayan's
  `flpsi_match` sub-sampler): 2-of-2 → TAR 95.4%, **0 false accepts / 372** — consistent with the
  analytical 8.2e-4.

## 4. Recommendation
- **Bridge:** plain Super-Bit 128-bit (no whitening) on the 224 model — single-finger EER 1.88%.
- **Deploy multi-finger fusion (2-of-3)** as the operating point: **TAR 99.9% @ FAR 2.4e-3**, or
  **3-of-3** for FAR ~2e-5. Per-finger flash-psi point `weight=14, t=2, T=64`; fuse client-side.
- This is all through the real flash-psi decision model (closed-form `q(H)=C(d−H,w)/C(d,w)` + Binomial,
  which is exact for the sub-sampling protocol in `flash-psi/src/subsample.rs`), no HE, linear scaling.

## Method note
The accept model is exact, not a heuristic: one mask is `weight` random positions, a sub-sample
matches iff both codes agree on all of them (prob `q(H)`), and the parties accept iff ≥ `t` of `T`
sub-samples match — so per-pair accept prob = `P(Binomial(T, q(H)) ≥ t)`. Fusion assumes finger
independence (justified: different fingers are independent ridge patterns), corroborated by the
empirical subject-grouped check.
