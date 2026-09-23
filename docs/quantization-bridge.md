# C2 - Binary Quantisation: bridging CNN embeddings and Hamming-distance PSI

*NeuralPSI contribution **C2**: "a binary quantisation scheme bridging floating-point CNN embeddings
and Hamming-distance PSI." This document explains the problem, the method we chose and why, the
results, and the non-obvious findings - everything needed to present and defend C2.*

> **One-line summary.** We map the CNN's 16 floating-point numbers to a **128-bit binary code** with a
> fixed, public *centre → (optional whiten) → Super-Bit projection → median-balanced sign* pipeline, so
> that **Hamming distance tracks identity**. On the properly trained model this code reaches **1.88% EER**
> - within ~1.5 pp of the floating-point ceiling (0.33%) that homomorphic encryption matches on - while
> being exactly the binary, balanced, near-i.i.d. format the flash-psi matcher needs.

---

## 1. Why C2 exists - the gap it bridges

NeuralPSI replaces Blind-Touch's homomorphic-encryption matching with **Fuzzy-Labelled PSI** (flash-psi).
That swap creates a representation mismatch:

| | Produces / consumes |
|---|---|
| **CNN feature extractor** (client) | a **16-D floating-point** embedding `e ∈ ℝ¹⁶`; similarity is **cosine / Euclidean** |
| **flash-psi matcher** (server) | **fixed-length binary strings**; similarity is **Hamming distance** with a sub-sampling threshold |

C2 is the converter in between. It must turn `e` into a binary code `b ∈ {0,1}¹²⁸` such that **two
prints of the same finger land at small Hamming distance, and different fingers land far apart** - and
it must do so in a form the PSI protocol (and the security argument) can actually use.

```
fingerprint ──CNN──► e ∈ ℝ¹⁶  ──[ C2 quantiser ]──►  b ∈ {0,1}¹²⁸  ──flash-psi──► match / no-match
              (floats)                (this doc)         (Hamming)
```

---

## 2. What a good bridge must satisfy (the requirements)

C2 is not just "round the floats." A usable bridge has to satisfy **four** constraints simultaneously,
and most off-the-shelf hashing schemes violate at least one:

1. **Similarity preservation** - Hamming distance must track identity (small for genuine, large for impostor).
2. **Fixed binary format** - exactly `d = 128` bits, native Hamming (flash-psi requirement).
3. **FLPSI-friendly error profile** - the matcher sub-samples bit positions, so bit errors should be
   **near-i.i.d. across positions** and bits should be **balanced (~50/50)**. Codes whose errors cluster
   in a few positions, or whose bits are lopsided, break the protocol's FPR/FNR analysis.
4. **Public, one-way-ish, cheap map** - the transform is published (like batch-norm stats); inference must
   be a single matmul + threshold; and it must not *trivially* expose the biometric (more in §8).

---

## 3. The methods we evaluated (and why we kept / rejected each)

We ran a multi-agent literature + design search, then validated empirically. Summary:

| Method | Idea | Verdict | Why |
|---|---|---|---|
| **Naïve `sign(>0)`** | threshold each float at 0 | ❌ baseline only | bits lopsided (impostor Hamming only ~4/16), codes collide, **EER 10-11%** |
| **ITQ** (Iterative Quantization) | learn a rotation `R`, then `sign((e−μ)R)`, 16-bit | ⚠️ better, but capped | recovers a lot over naïve sign (**EER ~7%**) but only **16 bits** of capacity |
| **Super-Bit LSH** | orthonormalised sign-random-projection, **128-bit** | ✅ **chosen** | best data-independent code; variance-reduced angle estimate; public seed |
| **Metric whitening** | rescale dims by `diag(√|w1|)` before projecting | ◑ situational | helps a *weak* model, **hurts a well-trained one** (see §7) |
| **Median balancing** | per-bit median threshold instead of 0 | ✅ always | guarantees ~50/50 bits → FLPSI-friendly |
| End-to-end / deep hashing | train the CNN to emit the code | ❌ rejected | heteroscedastic, correlated bits **break FLPSI's i.i.d. error model**; and ArcFace **collapsed** in 16-D (§7) |
| Multi-bit / Manhattan / Gray | spend several bits per dim to encode magnitude | ❌ rejected | a one-level error flips a **contiguous block** of bits → correlated errors break the sub-sampling model |
| p-stable / E2LSH | Euclidean bucket hashing | ❌ rejected | integer buckets, not balanced bits; no gain on the unit sphere |

**Headline rationale:** at a 128-bit code from a 16-D source, *data-dependent* learned hashing gives only
marginal gains (the source is information-limited), while the *data-independent* Super-Bit code is cheaper,
keeps the clean i.i.d. error profile FLPSI needs, and keeps the security story simple. The big accuracy
lever turned out to be the **embedding** (training), not the quantiser.

---

## 4. The chosen scheme - Super-Bit pipeline

Every stage is a **fixed, public, affine-then-sign** map (no learned non-linearity). Given the embedding
`e ∈ ℝ¹⁶`:

```
0. CENTRE          e0 = e − μ            μ = public population mean (L2-normalised CNN features sit on a
                                          spherical cap, so origin hyperplanes would be unbalanced)

1. WHITEN (optional)  e1 = diag(√|w1|) · e0     fold the head's learned per-dimension weights so Hamming
                                                tracks the *discriminative* metric, not raw cosine
                                                - USE ONLY ON UNDER-TRAINED MODELS (see §7)

2. SUPER-BIT       z  = P · e1           P ∈ ℝ¹²⁸ˣ¹⁶ = 8 blocks of a 16×16 Gaussian matrix,
                                          Gram-Schmidt-orthonormalised within each block,
                                          generated from a PUBLISHED seed

3. BALANCED SIGN   b_k = 1 if z_k > τ_k else 0   τ_k = per-bit population median  →  b ∈ {0,1}¹²⁸
```

At inference the public map is pre-composed offline to **one matmul + a threshold compare**:
`P' = P · diag(√|w1|)`, then `z = P'·(e − μ)`, `b = (z > τ)`. The constants `P', μ, τ` (+ the seed) are all
public and shipped beside the model.

### Why it works - the maths (random-hyperplane LSH)
For two unit vectors at angle `θ`, a random hyperplane `p` separates them with probability `θ/π`. So for a
`d`-bit sign-random-projection code,

```
E[ Hamming(b₁, b₂) ] / d  =  θ / π  =  arccos( cos-sim(e₁, e₂) ) / π
```

Genuine pairs (small angle) ⇒ small Hamming; impostors (large angle) ⇒ large Hamming. This is the exact
property flash-psi consumes.

### Why **Super-Bit** rather than plain SimHash
Plain SimHash uses i.i.d. Gaussian projection rows; **Super-Bit** orthonormalises the rows in blocks of
`N = 16` (= input dim). This keeps the estimator **unbiased** but gives it **strictly lower variance** for
the relevant (acute) genuine angles - i.e. a **tighter, more reliable** Hamming separation at the *same*
128 bits, for *zero* extra cost (the matrix is public and built once from a seed). It degenerates to plain
SimHash at block size 1, so it strictly subsumes the baseline.

---

## 5. How the code plugs into flash-psi

The quantiser was designed to satisfy the matcher exactly:

- **Length & metric:** outputs exactly `d = 128` bits, native Hamming - no adapter needed.
- **Balanced bits:** median thresholding makes each bit ~50/50 (measured **50.2%**), which the masked-OPRF
  and the sub-sampling statistics assume.
- **Near-i.i.d. errors:** each Super-Bit position flips roughly independently with probability `θ/π`, so the
  flash-psi sub-sampling FPR/FNR analysis (which depends on *total* Hamming distance) holds.
- **One-shot inference:** a single 128×16 matmul + threshold - negligible client cost.

---

## 6. Results - the EER ladder

Open-set protocol: held-out identities never seen in training (1,200 genuine + 240,000 impostor pairs),
SOCOFing, enroll = Real print, genuine probe = the finger's Altered-Easy print.

| Representation | Bits | **96×96 / 6-epoch CNN** | **224×224 / 150-epoch CNN** |
|---|---|---|---|
| Float features *(what HE matches on - the ceiling)* | - | 2.50% | **0.33%** |
| `sign(>0)` | 16 | 10.86% | 10.12% |
| ITQ | 16 | 7.12% | 5.55% |
| **Super-Bit** *(recommended)* | 128 | 5.14% | **1.88%** |
| Super-Bit + whitening | 128 | 3.91% | 2.98% |

Reading the table:
- **Super-Bit (128-bit) ≫ ITQ (16-bit) ≫ naïve sign.** Expanding to a *true* 128-bit code by random
  projection (not bit-repetition) carries far more discriminative information.
- On the well-trained model, **plain Super-Bit = 1.88% EER**, only **+1.55 pp** above the 0.33% float
  ceiling. Genuine Hamming ≈ 8/128, impostor ≈ 64/128, bit-balance 50.2%, separation d′ = 3.78.
- The binarisation "cost" (binary − float) fell from ~8 pp (naïve) to **~1.5 pp** (Super-Bit).

---

## 7. Non-obvious findings (the parts to highlight in the meeting)

1. **The embedding is the bottleneck, not the quantiser.** Retraining the CNN at full resolution
   (224/150) dropped the float ceiling 2.50% → **0.33%** and the Super-Bit code 5.14% → **1.88%**. The
   quantiser was already near-optimal; *fixing the features* is what moved the needle.

2. **Whitening reverses sign with model quality.** On the *weak* model it **helped** (5.14% → 3.91%) by
   reweighting noisy, uneven dimensions; on the *good* model it **hurt** (1.88% → 2.98%), because the
   embedding is already well-conditioned and the recovered `w1` (a noisier estimate) distorts good
   geometry. **Recommendation: plain Super-Bit, no whitening, on a properly trained model** - which also
   removes a published-`w1` dependency and simplifies the security story.

3. **The 16-D information ceiling is real.** 128 sign bits of a 16-D vector carry at most ~16 bits of
   genuine information; the residual ~1.5 pp binary-vs-float gap is **information-theoretic** and cannot be
   closed by a cleverer quantiser - only by a wider embedding (diminishing returns) or better features.

4. **ArcFace/CosFace collapses here.** A margin-classification head (suggested by the literature) **failed
   empirically**: 4,800 finger-classes cannot live as prototypes in a 16-D embedding (face ArcFace uses
   512-D), so training stalled at the trivial-guess loss. The faithful **Siamese pairwise loss** is the
   right trainer. *(This is an embedding-training finding, but it directly shaped C2's assumptions.)*

5. **Balanced, near-i.i.d. bits matter for the protocol, not just accuracy.** Median balancing isn't an
   accuracy trick (the arccos law is balance-independent) - it's what keeps the flash-psi sub-sampling
   statistics valid.

---

## 8. Security note on C2 (important correction)

The draft paper justifies privacy by claiming the sign map `b = sign(P·e)` is **one-way** because "128 bits
from 16 dimensions is underdetermined." **This is backwards.** 128 sign measurements of a 16-D vector is
**over-determined** (~8× oversampling), so **1-bit compressed sensing recovers the *direction* of `e`**
from public `P` and `b` (magnitude is lost but irrelevant under L2-normalisation).

**Consequence for C2:** the quantiser is *not* a privacy mechanism. Privacy must rest on the
**FLPSI/OPRF layer keeping the code `b` hidden from the server** - the server never sees `b` in the clear.
The paper's §4.2/§5 wording must be corrected accordingly. (C2's job is *accuracy + format*, not secrecy.)

---

## 9. Implementation

`quantizer.py` - pure NumPy, no training:

- `NeuralPSIQuantizer.fit(calibration_embeddings, w1, config)` → estimates the public constants
  `(μ, P', τ)` and saves them (`mu.npy`, `P_public.npy`, `tau.npy`, `meta.json`).
- `transform(e)` → `uint8[128]` code = one matmul + per-bit threshold.
- `QuantizerConfig(center, whiten, superbit, balance)` toggles each stage - this is exactly the
  **ablation ladder** used to produce §6.
- Self-test (`python3 quantizer.py`) validates the pipeline on synthetic data (separation, 50/50 balance,
  save/load round-trip).

Reproduce the C2 numbers on the trained model:
```bash
python3 07_superbit_eer.py --img 224 --ckpt feature_model_224.pt           # Super-Bit EER 1.88%
python3 07_superbit_eer.py --img 224 --ckpt feature_model_224.pt --whiten --w1 data/head_w1_224.npy  # 2.98%
```

---

## 10. Limitations & open questions

- **Single dataset / easy protocol.** Numbers are on SOCOFing **Altered-Easy** (synthetic alterations);
  cross-sensor / cross-session capture would be harder - a fair-evaluation TODO.
- **Information ceiling.** ~1.5 pp binary-vs-float gap is irreducible at 16-D; widening the embedding
  (Tier-3) trades storage for a small gain, with diminishing returns past ~64-128-D.
- **Real-crypto validation.** Correctness is verified via the faithful flash-psi Python port + the exact
  closed-form sub-sampling model; running our actual codes through the full OPRF/GC/VOLE/Shamir binary is
  outstanding.
- **Operating point is downstream of C2.** The single-finger code overlaps ~7% in the tails (driven by
  low-quality SOCOFing thumbs); the system-level answer (multi-finger fusion, operating-point re-tune) is a
  C1/C4 concern, not a C2 fix - but it's what turns 1.88% EER into a TAR 99.9% @ FAR 2.4e-3 deployment.

---

## TL;DR for the slide
- **Problem:** CNN gives 16 floats; flash-psi needs a 128-bit Hamming code. C2 is the bridge.
- **Solution:** centre → (optional whiten) → **Super-Bit** orthonormalised random projection → median sign
  → 128 balanced bits. Public, one matmul, `Hamming ≈ (d/π)·arccos(cos-sim)`.
- **Result:** **1.88% EER**, +1.5 pp from the 0.33% HE-accuracy ceiling; balanced, FLPSI-ready bits.
- **Findings:** embedding > quantiser; whitening helps weak/​hurts good models; 16-D is a hard ceiling;
  the sign map is *not* one-way (privacy comes from the OPRF).
