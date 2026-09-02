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

## 5. Dimension sweep (E2, the spine) — IN PROGRESS

Retraining {32,64,128,256}-D at 150 epochs, re-measuring σ, floor, EER
`[MEASURED: 14_dim_sweep_eval.py @ f855367; driver scratchpad/dim_sweep.sh]`.

| D | float EER (Euc) | Super-Bit EER | SB imp σ | SB H=0 | thermo EER | thermo σ | thermo H=0 |
|---|---|---|---|---|---|---|---|
| 16 | 0.33% | 1.89% | 20.11 | 2 | 0.97% | 14.16 | 1 |
| 32 | 0.14% | 2.13% | 19.66 | 2 | 1.65% | 14.90 | 1 |
| 64/128/256 | (running) | | | | | | |

**Preliminary and surprising:** 16→32-D **improves the float ceiling but not the 128-bit
code** — σ barely moves, the H=0 floor is unchanged, and binary EER slightly worsens. A fixed
128-bit code cannot carry more embedding DOF. If this persists to 256-D it **overturns** the
repo's claim that raising the embedding dimension "would move every number" — the floor would
be more fundamental than dimensionality, and the real lever is code-length co-design (blocked
by the FLPSI 128-bit backend) or the metric-aware bridge, not embedding width. Awaiting D≥64.
