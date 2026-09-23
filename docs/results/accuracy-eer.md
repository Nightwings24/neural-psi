# Tier-1 results - 224×224 / 150-epoch retrain (EXECUTED on GPU)

**Outcome: the embedding was the bottleneck. Retraining at full resolution lifted the float ceiling
7.5× and the Super-Bit binary EER 2.7×.** Run on an RTX 4050 (driver 595.80, torch 2.6.0+cu124).

## EER ladder - before vs after (held-out: 1,200 genuine / 240,000 impostor pairs)

| Representation | 96px / 6-epoch (CPU) | **224px / 150-epoch (GPU)** | Δ |
|---|---|---|---|
| **Float (HE accuracy ceiling)** | 2.50% | **0.33%** | −2.17 |
| `sign(>0)` 16-bit | 10.86% | 10.12% | −0.74 |
| ITQ 16-bit | 7.12% | 5.55% | −1.57 |
| **Super-Bit 128-bit** *(recommended)* | 5.14% | **1.88%** | **−3.26** |
| Super-Bit + metric whitening | 3.91% | 2.98% | - |

- Training converged to `pair_acc = 1.000`. Float EER **0.33%** is even better than the published
  Blind-Touch (~0.7% on SOKOTO) - confirming the prior 2.50% ceiling was an under-training artefact.
- **Best binary representation = plain Super-Bit 128-bit at 1.88% EER** (genuine Hamming 8.3/128,
  impostor 64.1/128, d′ = 3.78, bit-balance 50.2%).
- Remaining gap to float = +1.55 pp - the irreducible 16-D → 128-bit quantisation loss, now on a far
  lower base.

## Key finding: drop the whitening on a well-trained model
On the **weak 96px** model, metric whitening *helped* (5.14% → 3.91%) - it reweighted noisy,
uneven embedding dimensions. On the **224 model it HURTS** (1.88% → 2.98%): the embedding is already
well-conditioned, and the post-hoc `w1` (logistic-regression, train-acc 0.95) is a noisier estimate
that distorts good geometry. **Recommendation: use plain Super-Bit (center + Super-Bit projection +
median balance), no whitening.** Bonus - this removes the published-`w1` dependency and simplifies the
security argument.

## Another empirical correction: ArcFace collapses here
The improvement-search roadmap put an **ArcFace/CosFace margin head** in Tier-1. In practice it
**collapsed** - loss stuck at ≈ln(4800) with 0 accuracy - because it asks **4,800 finger-classes to
live as prototypes in a 16-D embedding**, which is geometrically impossible (face ArcFace uses 512-D).
The faithful **Siamese pairwise-BCE** loss (what Blind-Touch actually uses) trains cleanly in 16-D and
produced the numbers above. `train_arcface_224.py` is kept for reference but **use `train_gpu_224.py`**.

## How it was produced (all on GPU)
```bash
# 1. 224px arrays from the unzipped dataset/SOCOFing/ (no Kaggle zip needed)
python3 01b_extract_dir.py --img 224
# 2. faithful Blind-Touch Siamese-BCE, 150 epochs (auto CUDA+AMP)
python3 train_gpu_224.py --img 224 --epochs 150 --batch 64
# 3. measure Super-Bit EER on held-out fingers (GPU-accelerated embedding)
python3 07_superbit_eer.py --img 224 --ckpt feature_model_224.pt          # -> 1.88%  (best)
python3 07_superbit_eer.py --img 224 --ckpt feature_model_224.pt --whiten --w1 data/head_w1_224.npy  # 2.98%
```
The eval scripts auto-use the GPU if `torch.cuda.is_available()`, else CPU (backward compatible).

## What's next
- **Use plain Super-Bit 128-bit @ 1.88%** as the production bridge; re-tune the flash-psi `(weight, t)`
  operating point against the new (tighter) Hamming histograms.
- **Tier-2 (no GPU): multi-finger fusion** + operating-point re-tune to crush FAR further - still the
  highest end-to-end robustness lever, and now on a 1.88%-EER base.
- The 16-D → 128-bit gap (1.55 pp) is the only remaining binary-vs-float loss; closing it would need a
  wider embedding (Tier-3), with diminishing returns.

> **Confidence interval.** Identity-level bootstrap over all 1,200 x 1,199 held-out pairs
> gives **95% CI [1.56, 2.20]** for the Super-Bit 128-bit EER (`12_code_analysis.py`).
> That estimator's point value is 1.89% against the 1.88% below; the two differ by
> 0.01 pp, well inside the interval. Quote 1.88% with [1.56, 2.20].
