# NeuralPSI - Verification & Results Report

**Independent verification of the `implementation` Super-Bit pipeline, completion of the disabled
metric-whitening step, end-to-end flash-psi match + timing verification, and a researched
roadmap for what to do next.**

> TL;DR - Your headline numbers reproduce. Super-Bit (128-bit) gives **EER 5.14%** (you reported
> 5.16%), beating ITQ (7.12%) and naive sign (10.86%); the float ceiling is 2.50%. I also
> **enabled the metric whitening you had disabled** (by recovering the head weights `w1`), which
> takes Super-Bit to **EER 3.91%** - within **1.4 pp** of the float ceiling. The real flash-psi
> protocol builds and runs, and its timing is **linear at ~0.16-0.17 ms/record** (slightly faster
> than your 0.19 ms claim). The binarization is effectively done as a quantizer problem; the next
> gains are in the embedding (a 224×224/150-epoch GPU retrain) and at the PSI layer (multi-finger
> fusion).

---

## 1. System under test

```
CLIENT:  fingerprint → Blind-Touch CNN → 16-D float embedding → quantizer → 128-bit Hamming code
SERVER:  enrolled DB of 128-bit codes → flash-psi fuzzy match (NO homomorphic encryption)
RESULT:  matched label, revealed privately
```

- **CNN:** Blind-Touch Siamese port, real SOCOFing, 4,800 train / 1,200 held-out test identities,
  pair-accuracy 0.979. **Reduced scale: 96×96, 6 epochs, CPU** (see caveats §8).
- **Bridge (the contribution):** `sign(>0)` → ITQ → **Super-Bit LSH** (`quantizer.py`:
  center + Super-Bit projection + per-bit median balance, optional `diag(√|w1|)` whitening).
- **Matcher:** the real flash-psi protocol (kc1212/swanky: masked OPRF + garbled circuits +
  Vector-OLE + Shamir), code length d=128, T=64 subsamples, fuzzy threshold t=2.

---

## 2. Accuracy - EER ladder (held-out test: 1,200 genuine + 240,000 impostor pairs)

| Representation | Bits | **EER** | Gap to float | Hamming gen / imp |
|---|---|---|---|---|
| Float features *(HE matches on this)* | - | **2.50 %** | - (ceiling) | - |
| `sign(>0)` | 16 | 10.86 % | +8.36 | 0.59 / 3.79 (of 16) |
| ITQ | 16 | 7.12 % | +4.62 | 0.98 / 8.00 (of 16) |
| **Super-Bit** (no whitening) | 128 | **5.14 %** | +2.64 | 9.98 / 64.23 (of 128) |
| **Super-Bit + metric whitening** | 128 | **3.91 %** | **+1.41** | 7.47 / 64.12 (of 128) |

- Your reported Super-Bit EER (5.16%) reproduces at **5.14%** (seed/impostor-sampling noise).
- Bits are **perfectly balanced** (impostor Hamming = 64/128 = 50%); separation d′ ≈ 3.2.
- **Whitening (which you had disabled) closes another 1.23 pp** - see §4.

*Source of truth: `eer_report.txt` (included), produced by `07_superbit_eer.py`.*

---

## 3. The missing measurement - `07_superbit_eer.py`

**Why it was needed:** `results.md` quotes the 5.16% Super-Bit EER, but `06_superbit_export.py`
only writes the m=50 demo codes - **no shipped script actually computes the full Super-Bit EER**.
`07_superbit_eer.py` closes that gap.

**Methodology** (same open-set protocol as `03_eval_eer.py`, so the ladder is apples-to-apples):
- Held-out identities only (never seen in training). Enroll = Real print embedding; genuine
  probe = the same finger's Altered-Easy print; impostors = 200 random other identities per probe.
- Float score = Euclidean distance; binary scores = Hamming distance. EER = point where FAR = FRR.
- The Super-Bit quantizer is **fit on TRAIN-identity embeddings only** (no test leakage), exactly
  as `06_superbit_export.py` does. ITQ likewise fit on train only.
- Run: `python3 07_superbit_eer.py` → appends `eer_superbit_*` rows to `data/eer_report.txt`.

---

## 4. Completing the disabled feature - metric whitening (`recover_head_w1.py`)

Your `results.md` §5 caveat: *"Super-Bit metric whitening (`diag(√|w1|)`) was disabled (CNN head
weights not saved)."* The whitening only needs the **16 per-dimension weights `w1`** of the
Blind-Touch matching head, whose score is `sigmoid( Σₖ w1ₖ·(e_q,k − e_db,k)² + b )`.

**Methodology:** those 16 numbers can be recovered **without** the original head - fit a logistic
regression on the **squared differences** of train-identity embedding pairs (genuine=1 / impostor=0).
The learned coefficients *are* `w1` (negative: larger squared diff → lower match prob). This is
exactly the squared-distance metric that `diag(√|w1|)` whitening assumes.

- `recover_head_w1.py` → `head_w1.npy`. Logistic-regression train accuracy 0.992; `|w1|` ranged
  0.0028 … 0.615 (one dominant discriminative dimension).
- Re-running with whitening: `python3 07_superbit_eer.py --whiten --w1 data/head_w1.npy`
  → **EER 3.91%**, genuine Hamming tightened **9.98 → 7.47 / 128** (exactly your predicted
  "enabling it should tighten genuine Hamming further"), impostor unchanged at ~64/128.

**Caveat (important for the PSI layer):** whitening tightens genuine pairs but also pulls the
nearest impostor DB rows closer, so it **shifts the whole Hamming distribution** - the fixed
flash-psi operating point (w=24, t=2) must be **re-tuned** afterward (see §5). On the proper
224/150 model with the *true* squared-distance head, whitening should be cleaner than this
weak-model recovery.

---

## 5. End-to-end match verification - `run_flpsi_match.py`

We don't need the Rust binary to confirm the *match decision*: `flpsi_match.SubSampler` (your own
faithful port of `flash-psi/src/subsample.rs`) computes the identical accept/reject logic.
`run_flpsi_match.py` runs it across 10 random mask seeds for robustness.

**Super-Bit codes (`data/psi_sb/`, w=24, t=2, T=64):**
- **Genuine** → correct row 7 matched in **10/10 seeds**. (A near-neighbour row 40 at HD 15 vs
  row 7's HD 13 co-matches in 7/10 seeds - a **borderline co-match**, see below.)
- **Impostor** → **rejected in 10/10 seeds** (min HD 29, far from all rows).
- So the "{7} only" claim holds in some seeds; the genuine answer always *includes* the right row
  and the impostor is always rejected. The extra co-match is the weak-model embedding tail.

**Naive sign codes (for contrast, w=14, t=3):** collapse - genuine matches ~30 rows, impostor
yields ~29 false accepts (sign bits are lopsided). Confirms ITQ/Super-Bit are the usable bridges.

**On whitened codes:** the tighter distribution makes w=24/t=2 mismatched (extra co-matches + an
impostor leak) until the weight is raised (≥42 restores clean impostor rejection); the genuine
near-neighbour co-matches persist - an **embedding-quality** limit (96×96/6-epoch), not a bridge
bug. *Note: "return only the top-1 label" is NOT a valid fix - the FLPSI receiver never sees
per-row Hamming/sub-match counts, so it cannot rank rows (verified against the protocol source).*

---

## 6. flash-psi timing - VERIFIED (real Rust protocol)

Built kc1212/flash-psi (`cargo build --release`, 6m52s) and ran the real FLPSI simulation locally
(both parties over a socket pair; genuine masked-OPRF/GC/VOLE/Shamir). `fpr = 0` in the dummy
correctness check.

| DB size m | Your `results.md` | **Measured here** | per-record |
|---|---|---|---|
| 1,000 | 0.22 s | **0.195 s** | 0.195 ms |
| 5,000 | 0.93 s | **0.836 s** | 0.167 ms |
| 10,000 | 1.94 s | **1.67 s** | 0.167 ms |
| 50,000 | 9.33 s | **8.60 s** | 0.172 ms |
| 100,000 | 19.37 s | **16.21 s** | 0.162 ms |
| 1,000,000 | ~194 s (est) | **~162 s (est)** | 0.162 ms |

**Your central claim - linear scaling - is confirmed**: per-record is flat at **0.16-0.17 ms**
across 1k→100k (4 orders of magnitude). Absolute numbers are ~15% *faster* here (different
hardware/threads). Params: w=14, T=64, t=2. (flash-psi cost explodes with `t` via the Shamir
`C(t+1)` step - keep t=2, tune `weight`.)

**Not verified** (out of scope / not measured): the communication/storage numbers (need
`--track-io`), and the HE comparison row (quoted from the Blind-Touch paper, their 4-server
cluster - not reproducible here).

---

## 7. Conclusions

1. **Your pipeline reproduces.** EER ladder, the 5.16% Super-Bit headline, the genuine-matches /
   impostor-rejected decision, and linear timing all check out.
2. **Super-Bit (128-bit) clearly beats ITQ (16-bit): 5.14% vs 7.12% EER**, and **enabling whitening
   reaches 3.91%** - within 1.4 pp of the float ceiling.
3. **Binarization is effectively done as a quantizer problem.** The residual 1.4 pp gap is mostly
   irreducible 16-D→128-bit information loss; further quantizer tweaks (longer codes, fancier
   rotations, reliability weighting) are each ≤0.5 pp and capped by the float ceiling.
4. **Security note for the paper:** the §4.2 justification "128 bits from 16 dims is underdetermined
   ⇒ one-way" is backwards - it is *over*-determined, and 1-bit compressed sensing recovers the
   *direction* of `e`. Privacy must rest on the FLPSI/OPRF layer keeping the code hidden, not on
   irreversibility of the sign map. (Reword §4.2/§5.)

---

## 8. Honest caveats / limitations

- **Weak model:** 96×96, 6 epochs, CPU - not the paper's 224×224 / 150 epochs. The 2.50% float
  ceiling here is itself far above the published Blind-Touch (~0.7% EER on SOKOTO). The
  float-vs-binary *gap* is the trustworthy conclusion; absolute EER will improve on a GPU.
- **Match verified via the Python port**, not the Rust `fingerprint` binary (which needs your
  `fingerprint.rs` + `flpsi.rs` patch). The decision logic is identical; the crypto is not exercised.
- **Whitening `w1` was recovered** from the weak model via logistic regression, not the original
  head (which wasn't saved). On a 224 model that saves `w1`, whitening will be cleaner.
- **Timing** measured on a single desktop, single runs; communication numbers not measured.

---

## 9. Recommended next steps (from a researched, adversarially-verified search)

**Tier 1 - lift the float ceiling (biggest prize, needs a GPU):**
- **Train the real 224×224 / 150-epoch model + an ArcFace/CosFace margin head** (`train_gpu_224.py`
  already exists). Targets ~0.7% float EER (the published Blind-Touch number), lifts *both* float
  and binary EER, and **saves the true `w1`** so whitening is principled. The binding constraint is
  extractor *capacity* (resolution/epochs/loss), not embedding dimension.

**Tier 2 - cheap, no-GPU, no-retrain (harden the current system now):**
- **Multi-finger fusion at the PSI layer** (enroll K fingers as K independent FLPSI DBs, require a
  k-of-K quorum). Crushes FAR ~4-5 orders of magnitude (≈4.6e-6 → ~6e-11 for 2-of-3), t stays 2,
  scaling stays linear. Biggest end-to-end robustness win.
- **Re-tune `(weight, θ)`** jointly from the measured Hamming histograms toward a genuine-safe point
  - the correct fix for the co-match leakage (NOT top-1, which the protocol can't do).
- **Multi-sample enrollment** (average codes per finger) - real but bounded (~half the genuine
  noise is reducible), no retrain.

**Tier 3 - only when already retraining:** widen the embedding 16→64-D (modest; flat below ~512-D);
end-to-end hash-aware head (balance+decorrelation+margin) - flash-psi-compatible but gain unproven;
bundle into the 224 retrain, don't do standalone.

**Stop doing:** standalone quantizer micro-optimization - diminishing returns, all capped by the
float ceiling.

---

## 10. Files in this handoff & how to run

*(These files are snapshot copies in `reports/snapshot/`; the live versions are in `implementation/week1/`.)*

| File | What it is | Where it goes |
|---|---|---|
| `07_superbit_eer.py` | **NEW** - full Super-Bit EER eval (the missing measurement) | `implementation/week1/` |
| `recover_head_w1.py` | **NEW** - recover head `w1` to enable metric whitening | `implementation/week1/` |
| `head_w1.npy` | recovered 16-D head weights (regenerable) | `implementation/week1/data/` |
| `run_flpsi_match.py` | **NEW** - pure-Python flash-psi match verifier (no Rust) | `implementation/week1/` (next to `flpsi_match.py`) |
| `eer_report.txt` | results incl. the appended Super-Bit rows | reference |
| `report.md` | this document | reference |

```bash
# from implementation/week1/  (needs the trained model + data already present there)
python3 07_superbit_eer.py                              # -> Super-Bit EER 5.14%
python3 recover_head_w1.py                              # -> data/head_w1.npy
python3 07_superbit_eer.py --whiten --w1 data/head_w1.npy   # -> Super-Bit EER 3.91%
python3 06_superbit_export.py                           # -> data/psi_sb/{genuine,impostor}.txt
python3 run_flpsi_match.py                              # -> genuine row 7 always, impostor rejected

# real flash-psi timing (needs Rust; flash-psi cloned into implementation/flash-psi)
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env
#   cd ../flash-psi && cargo build --release
python3 benchmark_timing.py --max 100000               # -> ~0.16-0.17 ms/record, linear
```
