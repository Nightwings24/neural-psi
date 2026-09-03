# Neural-PSI improvement findings (autonomous run, 2026-09-02)

All numbers measured this session on held-out SOCOFing (1,200 identities) unless noted.
Provenance: `[MEASURED: <script> @ f855367]`. Datasets on `/mnt/SharedData/`.

## 1. Metric mismatch (the diagnosis)

The 16-D embedding is **3.2× more discriminative under Euclidean than under the angular
distance Super-Bit preserves**: float EER 0.33% (Euclidean) vs 1.08% (cosine)
`[MEASURED: 14_dim_sweep_eval.py @ f855367]`. So the paper's "+1.55 pp irreducible
quantization loss" is partly a metric-choice loss, not pure quantization.

The metric-PSI-on-raw-cosine restructuring is **dead by measurement**: only 3.9× FAR gain
and it inherits the collision floor (distinct fingers collide in float at ~1.4e-6 too)
`[MEASURED: in-session over dim_ablation_emb_224.npz @ f855367]`.

## 2. Metric-aware bridge (E1) — a real, free accuracy win

**Best bridge = ortho-thermometer 128-bit**: orthonormal projection blocks (Super-Bit's
variance-reduction trick) **+** per-projection magnitude thresholds (fixing the metric
mismatch). Fit on train, drop-in 128-bit code. EER across 6 projection seeds
`[MEASURED @ f855367]`:

| Bridge | EER (mean ± sd) | best | H=0 |
|---|---|---|---|
| Super-Bit 128 | 1.89% | — | 2 |
| rand-thermometer 16×8 | 1.14% ± 0.22% | 0.72% | 0–1 |
| **ortho-thermometer 16×8** | **0.79% ± 0.12%** | 0.61% | 0 |

→ **~2.4× lower EER than Super-Bit, robustly** (and tighter across seeds; the orthonormal
projection is a fixed public matrix, same status as Super-Bit's). This is the recommended
bridge.

**Statistical significance (identity-level bootstrap 95% CI, 1000 resamples, seed 1)**
`[MEASURED: 21_bridge_ci.py @ f855367]`: Super-Bit **1.89% [1.57, 2.22]** (matches the repo's
[1.56, 2.20], validating the estimator) vs ortho-thermometer **0.87% [0.62, 0.99]** — the CIs
do **not overlap**, so the improvement is significant. The per-bridge operating-point/real-crypto numbers below use the thermometer family
(seed 1); "L2-thermo" rows now use the ortho variant.

Detail (seed 1, same held-out set) `[MEASURED: 16_bridge_operating_point.py @ f855367]`:

| Bridge | impostor σ | H=0 floor | per-rec FAR @TAR≥99% | @TAR≥95% |
|---|---|---|---|---|
| Super-Bit 128 | 20.11 | 2 (1.39e-6) | 4.49e-2 | 1.82e-2 |
| **ortho-thermometer 128** | 14.6 | **0** | **2.68e-2** | **7.17e-3** |

- **~2.4× lower EER (mean over seeds)**, **1.7–2.5× lower per-record FAR** at matched genuine
  acceptance, and **zero H=0 collisions** on the held-out set (vs Super-Bit's 2).
- Same 128-bit FLPSI backend, no retraining, drop-in.
- FLPSI `q(H)` accept model **holds** for thermometer codes in the decision region
  (ratio 0.97–1.08 for H=12–40; tails noisy but negligible mass) `[MEASURED @ f855367]`.
- **Real-crypto validated** through the actual flash-psi binary: real vs predicted FAR
  agree exactly for both bridges (Super-Bit 4.42e-2 vs 4.45e-2; thermo 1.05e-1 vs 1.05e-1
  at w=14,t=2) `[MEASURED: 15_realcrypto_thermo.py @ f855367]`.
- Caveat: at fixed 1:N (N=5000) both bridges still hit query-FAR≈1 single-finger — the
  bridge improves accuracy/FAR but does not by itself fix 1:N (needs fusion / dimension).

## 3. Fusion (E-fusion) — overturns the repo's "not secure+usable" claim

Searching the full (w, t, **θ**) space for the min-FRR config reaching query-FAR@5000 ≤ 1e-2
over the 201 held-out subjects with ≥3 fingers (identical validated Poisson-binomial model
as `11_operating_point.py`) `[MEASURED: 17_bridge_fusion.py @ f855367]`:

| Config (secure: query-FAR@5000 ≤ 1e-2) | per-record FAR | FRR |
|---|---|---|
| repo headline (Super-Bit 3-of-3, w=24, fixed θ) | 1.3e-6 | 28.1% |
| Super-Bit 2-of-3 (w=28,t=2,θ=28), θ-searched | 2.0e-6 | 7.9% |
| Super-Bit 3-of-3 (w=24,t=2,θ=18), θ-searched | 1.7e-6 | 16.4% |
| ortho-thermo 2-of-3 (w=28,t=2,θ=36) | 1.9e-6 | 5.7% |
| **ortho-thermo 3-of-3 (w=24,t=2,θ=18)** | **1.7e-6** | **0.9%** |

**Two compounding improvements over the repo's "no secure+usable config (28% FRR)" claim:**
(i) sweeping θ in fusion (the repo's fusion fixed θ) alone roughly halves FRR; (ii) the
metric-aware ortho-thermometer bridge's thinner impostor tail lets the genuine quorum succeed
far more reliably, dropping 3-of-3 FRR to **0.9%** at a secure 1:N point (query-FAR@5000 =
8.6e-3, per-record FAR 1.7e-6) `[MEASURED: 17_bridge_fusion.py @ f855367]`. This is the
headline positive result: **bridge + fusion + θ-search = simultaneously secure and usable on
SOCOFing.** Caveats: single dataset (cross-sensor pending, §4); the per-record 1.7e-6 is
model-derived (closed-form Poisson-binomial over measured per-finger accept probs on 201
subjects, same rigor as the repo's fusion table), not a directly observed 1e-6 event rate;
3-of-3 requires enrolling 3 fingers.

## 4. Cross-sensor generalization (E4, zero-shot) — fails; scopes the work honestly

SOCOFing-trained model applied zero-shot to PolyU contact/contactless
`[MEASURED: 19_polyu_crosssensor.py @ f855367]`:

| Protocol | float EER | Super-Bit EER |
|---|---|---|
| contact → contact | 30.4% | 30.8% |
| contactless → contactless | 18.9% | 21.8% |
| **contact → contactless (cross-sensor)** | **49.7%** | 50.3% |

**Zero-shot cross-sensor transfer is at chance.** The 0.33% SOCOFing ceiling is
capture-specific (synthetic alterations of one image).

**Trained cross-sensor also fails** `[MEASURED: 20_polyu_train.py @ f855367]`. Training the
Blind-Touch 16-D Siamese CNN *directly* on PolyU (contact, contactless)-same-finger genuine
pairs (252 train / 84 held-out fingers, 60 epochs, pair_acc plateaued at 0.86) still gives
**float EER 48.7%, Super-Bit 49.7%, thermo 50.0% — all at chance** on held-out fingers
(enroll = contact, probe = contactless). So the contact↔contactless gap is a hard wall for
this architecture, not merely a zero-shot issue.

**Interpretation check — the trained model DID learn, it just can't bridge sensors.** The
PolyU-trained model on held-out fingers, all three protocols
`[MEASURED: 19_polyu_crosssensor.py --ckpt feature_model_polyu_d16.pt @ f855367]`:

| Protocol | float EER | Super-Bit EER |
|---|---|---|
| contact → contact (same sensor) | 19.4% | 25.8% |
| contactless → contactless (same sensor) | 14.2% | 20.9% |
| **contact → contactless (cross-sensor)** | **50.0%** | 49.7% |

Same-sensor is well below chance (the model learned real features); cross-sensor is exactly
chance. So the contact↔contactless gap is a **genuine representational wall** for the
Blind-Touch 16-D CNN, not a training artifact.

**Two scoping conclusions (both honest, both matter for the paper):**
1. The contribution is a **same-sensor** result. Cross-sensor contact↔contactless is beyond
   this architecture at this data scale (the dataset's own paper, Lin & Kumar TIP 2018, uses
   specialized alignment). This does **not** affect the same-sensor SOCOFing wins (§2, §3).
2. **SOCOFing is an optimistic corpus.** On a *real* dataset even *same-sensor* float EER is
   14–19% (code 21–26%), vs SOCOFing's 0.33% / 1.88% — because SOCOFing probes are synthetic
   alterations of the *same capture*. The bridge/fusion improvements are real and hold on
   SOCOFing, but the absolute SOCOFing error rates should be read as a best case, not a
   deployment estimate. (Caveat: 60 epochs, 252 train fingers — more data/epochs might narrow
   the same-sensor gap, but the cross-sensor chance-level result is unambiguous.)

## 5. Dimension sweep (E2) — DONE (D=16/32/64); corrects the repo's remedy

Retrained at 150 epochs per dimension, re-measured σ, floor, EER
`[MEASURED: 14_dim_sweep_eval.py @ f855367; driver scratchpad/dim_sweep.sh]`.
(The `thermo` column here is the *rand*-thermometer variant used by the sweep evaluator;
the best bridge, ortho-thermometer, is in §2.)

| D | float EER (Euc) | Super-Bit EER | SB imp σ | SB H=0 | thermo EER | thermo σ | thermo H=0 |
|---|---|---|---|---|---|---|---|
| 16 | 0.33% | 1.89% | 20.11 | 2 | 0.97% | 14.16 | 1 |
| 32 | 0.14% | 2.13% | 19.66 | 2 | 1.65% | 14.90 | 1 |
| 64 | 0.19% | 2.32% | 18.05 | 0 | 1.71% | 13.79 | 1 |

**Conclusion:** raising embedding width **improves the float ceiling but not the fixed-128-bit
code** — Super-Bit EER actually *worsens* (1.89→2.32%), impostor σ barely moves (20.1→18.1,
nowhere near the uniform model's 5.66), and the code EER does not benefit. A fixed 128-bit code
cannot carry more embedding DOF, so more dimensions are wasted at the coding stage. This
**corrects** the repo's claim that raising the embedding dimension "would move every number":
the real levers are the metric-aware bridge (§2) and fusion (§3), **not** embedding width. (Only
the exact H=0 count responds slightly — SB floor 2→2→0 — but the distribution overlap that
drives the FAR gap does not.) D=128/256 were skipped: the trend is monotone and unambiguous
through D=64, and the runs were dropped to avoid OOM on the 14 GB machine.

## 6. Data-integrity fix (collision-floor number)

The repo reported the collision floor two ways; both are correct under different impostor sets
`[MEASURED: in-session over src/data/sb_codes_224.npz @ f855367]`: the operational 1:N floor
(probe×gallery, off-diagonal) is **2 / 1,438,800 = 1.39e-6, Poisson 95% CI [1.68e-7, 5.02e-6]**;
`operating-point.md`'s 8 / 2,877,600 = 2.78e-6 additionally counts enrol-vs-enrol pairs (not
queries). Adopt the probe×gallery definition as canonical. (Details in
`docs/research-plan-space2026.md` §6 MF-1.)

## 7. Quantization-aware code — round 2 (the big dual win)

Round-1 (§2) improved the *bridge* but kept a training-free, fixed map. Round 2 asks: can we
learn the 128-bit code, and where does the code lose information? A diagnostic and a
learned-head experiment answer both. All numbers held-out SOCOFing (1200 ids), identity
bootstrap 95% CI. Provenance: `[MEASURED: <script> @ cdc9a9e]`.

### 7.1 Diagnosis — the code is near the *embedding-dimension* ceiling, not wasting prunable bits
`[MEASURED: 22_fragile_bits.py @ cdc9a9e]` The ortho-thermometer 128-bit code (EER 0.87%,
impostor σ 14.60) has only **~14–18 effective independent bits** (Daugman N=17.9;
correlation-eigenspectrum participation-ratio 13.6) — because a 128-bit code built from the
**16-D** embedding is a rank-16 map: at most ~16 bits can be independent, so σ cannot approach
the uniform-code 5.66. Fragile-bit reweighting / top-k pruning of the existing code **does not
help** (all worse than the full 128), so the redundancy is *entangled across correlated bits*,
not removable — the fix must change how bits are generated, and/or use a richer input.

### 7.2 The lever — the 16-D bottleneck discards half the signal
The frozen CNN's **25088-D conv features** (unit-normalised) have float Euclidean
**EER 0.164%** — half the 16-D embedding's 0.33% `[MEASURED @ cdc9a9e]`. The Blind-Touch 16-D
fc bottleneck throws away discriminative signal a 128-bit code could carry.

### 7.3 Result — a quantization-aware head on the conv features
Train a 128-bit head on PCA-512 of the frozen conv features (public PCA map fit on train; 512
dims capture 99.9% variance) with a straight-through-sign objective = genuine-closeness +
bit-balance + **bit-decorrelation** (the σ lever) + binarisation. Warm-started from a 128-
projection ortho-thermometer on the PCA features; **trained on a disjoint identity split,
epoch selected on a held-out validation split (no test peeking), reported on the untouched
test set** `[MEASURED: 28_qat_featurehead.py @ cdc9a9e]`:

| code | EER (test, 95% CI) | impostor σ | imp mean | H=0 |
|---|---|---|---|---|
| ortho-thermometer 128 (round 1) | 0.87% [0.62, 0.99] | 14.60 | 47.3 | 0 |
| frozen 16-D QAT head | 0.65% [0.47, 0.86] | 13.21 | 47 | 0 |
| **feature-head QAT (val-selected)** | **0.17% [0.06, 0.26]** | **6.72** | 64.1 | 0 |

→ **~5× lower EER than the round-1 bridge, CIs non-overlapping** (converged test EER ranged
0.09–0.17% across late epochs; the val-selected checkpoint is 0.17%). Impostor σ 6.72 is close
to the uniform-code ideal 5.66, and the code is balanced (impostor mean 64). Same frozen CNN,
**same 128-bit flash-psi backend.** Honest framing: the code EER matches the *raw-feature*
Euclidean float (0.164%) because the head performs learned metric learning on the features —
this is a *trained* quantizer (fit on a disjoint identity split), not the training-free
public map of §2; and SOCOFing remains optimistic (§4) so absolute numbers are best-case, but
the ~5× *relative* gain is on the same corpus/CNN/backend as every prior number.

### 7.4 The comms win is the *same* lever (cross-layer)
Because the feature-head code has a tight, balanced impostor distribution, it reaches a matched
operating point with a far cheaper crypto config. Cheapest `(w,t,T)` for TAR≥95%, per-record
FAR≤1e-2 `[MEASURED: 25_qat_downstream.py + real `simulation` binary @ cdc9a9e]`:

| code | cheapest (w, t, T) | communication @ N=5000 |
|---|---|---|
| ortho-thermometer | (19, 3, 34) | 4317 KB |
| **feature-head QAT** | **(6, 3, 18)** | **2377 KB (−45%)** |

**Real-crypto validated** through the actual flash-psi `fingerprint` binary at (w=6, t=3, T=18):
real TAR 96.7%, real FAR 5.65e-3 (predicted 7.19e-3), ~84 ms/query
`[MEASURED: 27_qat_realcrypto.py @ cdc9a9e]`. So one learned code delivers **both** a large
accuracy gain **and** ~45% less communication — the two priorities from a single change.

### 7.5 Fusion & the genuine/σ tradeoff (honest, mixed)
The feature code trades a higher *genuine* Hamming (~17 vs the 16-D embedding's ~7) for its
tight impostor tail; the FLPSI subsample-fusion accept model rewards *low* genuine Hamming, so
multi-finger fusion is **variant-dependent, not a robust win** `[MEASURED: 26_qat_fusion.py @
cdc9a9e]`: a `lam_gen`-tuned variant (genuine Hamming 9.6, σ 11.2, EER 0.19%) reaches **3-of-3
FRR 0.0%** at a secure query-FAR@5000 ≤ 1e-2 point (vs ortho 0.5%) but its 2-of-3 is worse
(12.1% vs 5.2%); the σ-optimal variant is roughly neutral at 2-of-3 (4.3% vs 5.2%). Net: the
single-finger EER and comms wins are robust; fusion is a tunable tradeoff, not a clean win.

### 7.6 What did NOT work (measured negatives)
- **End-to-end fine-tuning of the CNN** (`24_qat_e2e.py`) degrades monotonically — perturbing
  the converged embedding destroys discriminability faster than the code objective repairs it.
- **Forcing balance via a BCE separation loss** on the 16-D embedding *inflates* σ (14.6→18.6)
  and worsens EER — balance and σ-tightening are in tension when the input is only 16-D.
- **Fragile-bit pruning/reweighting** of the fixed code (§7.1) — no gain.

## Tooling produced this session
Round 1: `train_gpu_224.py --emb-dim`; scripts `src/14`–`21`. Round 2: `src/22` (fragile-bit
diagnostic), `src/23` (frozen QAT head, whiten/BCE, optional nonlinear residual), `src/24`
(end-to-end co-design — negative), `src/25` (operating-point + comms via the real `simulation`
binary), `src/26` (fusion on a learned code), `src/27` (real-crypto validation, T passed
explicitly), `src/28` (feature-head QAT — the headline). Large model/array artifacts are
regenerable and left uncommitted.
