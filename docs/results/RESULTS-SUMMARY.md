# Neural-PSI — results at a glance (one-page summary)

Companion to `improvement-findings.md`. Pipeline:
`fingerprint → CNN → embedding → binariser → 128-bit code → Fuzzy-Labelled PSI → accept/reject`.
Two improvement iterations, **same 128-bit FLPSI backend throughout**. All EER on held-out
SOCOFing (1,200 identities) with identity-bootstrap 95% CIs.

## The one-line story
The **binariser** (embedding → 128-bit code) was the weak link. Round 1 fixed the *distance
metric* it preserves; Round 2 fixed how it *uses its bits*. The code's impostor Hamming
distribution drives **both** accuracy and communication, so one better code improves both.

## Headline numbers

| Metric | Original (Super-Bit) | Round 1 (metric-aware bridge) | **Round 2 (quantization-aware head)** |
|---|---|---|---|
| EER — Altered-Easy | 1.89% [1.57, 2.22] | 0.87% [0.62, 0.99] | **0.17% [0.06, 0.26]** |
| EER — Altered-Medium | 4.48% | 2.61% [2.17, 2.92] | **0.84% [0.62, 1.17]** |
| EER — Altered-Hard | 7.04% | 4.10% [3.64, 4.60] | **2.18% [1.81, 2.70]** |
| Impostor Hamming σ (ideal 5.66) | 20.1 | 14.6 | **6.72** |
| Effective bits of 128 | ~10 | ~14–18 | **~90** |
| H=0 collisions | 2 | 0 | **0** |
| Communication @ N=5000 (matched op-point) | — | 4317 KB | **2377 KB (−45%)** |

**vs. the last committed version (Round 1): ~5× lower EER at Easy (≈3× Medium, ≈1.9× Hard),
all CIs non-overlapping, plus 45% less communication — from one code change.**

## What each iteration did

**Round 1 — metric-aware bridge.** The embedding is 3.2× more discriminative under Euclidean
(float EER 0.33%) than the cosine distance Super-Bit preserves (1.08%). Replaced Super-Bit with
an **ortho-thermometer** code (orthonormal projections + magnitude thresholds) that preserves
the right geometry. EER 1.89% → 0.87%, drop-in, no retraining. Also: fusion-threshold search
turned a claimed 28% FRR into **0.9% FRR** at a secure 1:N point.

**Round 2 — quantization-aware feature-head.** The 128-bit code used only ~14–18 real bits
(a 16-D embedding can't fill 128 bits), while the CNN's own 25,088-D features have float EER
**0.164%** — half the 16-D bottleneck's. Trained a **learned 128-bit code** on those features to
be **balanced + decorrelated** (toward the uniform code the protocol assumes). EER 0.87% → 0.17%,
σ 14.6 → 6.72, and −45% communication — **validated end-to-end through the real crypto binary**
(real TAR 96.7%, FAR 0.57%).

## Why accuracy and communication improved together (the key idea)
FLPSI accepts by randomly subsampling `w` of 128 bits `T` times; communication scales with `T`.
A code with a tighter, more uniform impostor distribution separates genuine from impostor with
**fewer subsamples**, so the same code that lowers EER also lowers `T` → less communication.

## Generality (honest scope)
Tested on PolyU and FVC2002 (and a pretrained DeepPrint backbone):
- **Universal:** the impostor-σ collapse (→ the communication benefit) holds on every corpus.
- **Gap-dependent:** the big accuracy win needs a large *quantization gap* (float ≪ code). It's
  dramatic on high-fidelity SOCOFing; marginal on harder real corpora where the **extractor**,
  not the code, is the bottleneck. Off-the-shelf backbones don't transfer across corpora
  (a known-hard problem), so a second dramatic accuracy result needs a large high-fidelity
  corpus — **NIST SD302** (request pending).

## Caveats to state plainly
- SOCOFing is optimistic (synthetic same-capture alterations); the ~5× is a *relative* gain on
  the same corpus/CNN/backend as every baseline number.
- Round 2's binariser is *trained* (on a disjoint identity split), vs Round 1's training-free map.
- Multi-finger fusion is a tunable tradeoff, not a robust win.

## What didn't work (measured negatives)
End-to-end CNN fine-tuning · BCE balance objective · fragile-bit pruning · pretrained-backbone
(DeepPrint) transfer. All documented in `improvement-findings.md` §7.7.

## Comparison with Blind-Touch (the prior neural approach, arXiv:2312.11575)
Blind-Touch is the HE-based system we build on (same Siamese CNN / 16-D embedding). It matches
on the *encrypted float* embedding via CKKS on a 3-server cluster; we match on a *binary code*
via 2-party Fuzzy-Labelled PSI. Numbers from their paper vs this work:

| | Blind-Touch (CKKS HE, 3 servers) | Neural-PSI (FLPSI, 2-party) — this work |
|---|---|---|
| SOCOFing EER | 0.7% | Round 1 0.87% · **Round 2 0.17%** |
| PolyU EER | 2.5% | 11.7% (not comparable — see note) |
| Communication @ N=5000 | **~856 KB** (compressed CKKS) | 7953 KB → **2377 KB** (Round 2) |
| Latency @ N=5000 | ~650 ms (3-server cluster) | ~84 ms/query real-crypto (small N; not directly comparable) |
| Trust / deployment | server does encrypted inference; needs a cluster | lightweight 2-party; reveals only the label on a match |

**Honest reading:**
- **Accuracy (SOCOFing):** we win — feature-head **0.17%** vs Blind-Touch **0.7%** (~4×), on the
  same corpus and CNN family.
- **Communication:** **Blind-Touch wins.** CKKS-with-compression (856 KB) is lower than our FLPSI
  (2377 KB after Round 2). Our contribution cut *our own* comms 45%, narrowing the gap from ~9×
  to ~2.8×, but PSI does not beat compressed HE on bandwidth. Frame our comms result as
  "−45% vs our own baseline," **not** as beating Blind-Touch.
- **PolyU:** not apples-to-apples — Blind-Touch trained a PolyU-specific extractor (2.5%); our
  generality test reused a cross-modal-trained CNN. This actually reinforces our Round-2 finding
  that on real corpora the *extractor*, not the code, is the bottleneck.
- **Our real edge over Blind-Touch:** better SOCOFing accuracy **and** a simpler 2-party protocol
  (no homomorphic-encryption cluster), with a different privacy model (label-only disclosure).

## Where it lives
Branch `space2026-metric-aware-bridge`. Scripts `src/14`–`32`. Full detail in
`improvement-findings.md` (§1–6 Round 1, §7 Round 2); next steps in `future-directions.md`.
