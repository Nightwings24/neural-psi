# What the 128-bit code actually contains

Reproduce with `python 12_code_analysis.py`. Raw output: `code-analysis.json`; figure:
`paper/figures/bit-dependence.pdf`.

`operating-point.md` shows the impostor Hamming distribution has 3.6× the standard deviation
the FLPSI error analysis assumes, and that this alone breaks the published parameters. This
document establishes *why*, puts confidence intervals on the headline accuracy, and answers the
bit-budget objection.

All measurements are on 1,200 held-out SOCOFing identities. The quantiser and the ITQ rotations
are fitted on the 4,800 training images only, disjoint from everything scored here.

---

## 1. The code is balanced - which is exactly why the problem is invisible

| statistic | value |
|---|---|
| per-bit balance (fraction of 1s) | 0.466 - 0.540 across all 128 bits |
| mean inter-bit \|ρ\| | **0.254** |
| max inter-bit \|ρ\| | 0.858 |
| bit pairs with \|ρ\| > 0.3 | **36.2%** |
| participation ratio of the bit correlation matrix | **9.84** (of 128) |
| eigenvalues carrying 90% of the variance | 55 |
| top eigenvalues | 24.5, 23.7, 17.2, 9.3, 7.9, 5.8, 2.1, 1.3 |

Every bit is close to balanced, so the mean impostor Hamming distance comes out at exactly
`d/2 = 64`, as the uniform model predicts. **A first-moment check therefore passes cleanly on a
code that violates the model badly**, which is what makes this failure mode worth publishing.

The participation ratio is the honest statement of the code's capacity: the 128 bits behave like
roughly **10 independent binary decisions**, not 128. That is expected - they are 128 projections
of a 16-dimensional embedding - and it is the mechanism behind the wide impostor distribution.

## 2. But the accept model itself is accurate

The FLPSI analysis assumes a sub-sample of `w` positions is error-free with the hypergeometric
probability `q(H) = C(d−H, w) / C(d, w)`. Since Super-Bit's advantage comes precisely from making
bits *dependent*, that assumption deserved a direct test. Measured on real code pairs at each
Hamming distance, with 200 random sub-sample draws per pair, `w = 14`:

| H | pairs | q̂ measured | q hypergeometric | ratio |
|---|---|---|---|---|
| 5 | 52 | 0.5516 | 0.5549 | 0.994 |
| 8 | 113 | 0.3882 | 0.3849 | 1.009 |
| 10 | 232 | 0.3005 | 0.3000 | 1.002 |
| 15 | 788 | 0.1555 | 0.1577 | 0.986 |
| 20 | 1,565 | 0.0777 | 0.0804 | 0.968 |
| 25 | 2,658 | 0.0391 | 0.0396 | 0.989 |
| 30 | 3,992 | 0.0190 | 0.0188 | 1.013 |
| 40 | 7,694 | 0.0034 | 0.0037 | 0.919 |

Over all 36 measured distances the ratio has **median 1.002 and range [0.919, 1.099]**.

**This is a positive result and it sharpens the diagnosis.** Bit dependence does *not* corrupt
the conditional accept model: given `H`, sub-sampling behaves hypergeometrically, because the
mask is a uniformly random set of positions and averages over which particular bits differ. The
dependence shows up in the distribution **of** `H`, not in `accept(H)`.

Two consequences:

1. The defect is isolated to one place - the impostor `H` distribution - rather than being spread
   through the analysis.
2. The closed-form error rates in `operating-point.md`, which combine the measured `H` histograms
   with the hypergeometric `accept(H)`, are therefore trustworthy. Both halves are validated
   independently.

## 3. Accuracy, with confidence intervals and at a matched bit budget

Comparing a 128-bit Super-Bit code against a 16-bit ITQ code is a bit-budget mismatch, so ITQ is
fitted at both 16 and 128 bits. Confidence intervals are 95% bootstrap over **identities**, not
over pairs - pairs sharing a finger are not independent observations, and resampling pairs
understates the interval.

| Coder | Altered-Easy | Altered-Medium | Altered-Hard | H=0 collisions (of ~1.44M) |
|---|---|---|---|---|
| **Super-Bit 128** | **1.89%** [1.56, 2.20] | **4.48%** [3.90, 4.94] | **7.04%** [6.31, 7.45] | **2** |
| ITQ 128 | 4.55% [3.91, 5.31] | 7.11% [6.39, 7.73] | 9.06% [8.47, 9.75] | 517 |
| ITQ 16 | 6.13% [5.54, 6.80] | 8.49% [7.84, 9.19] | 11.58% [10.56, 12.20] | 6,394 |

Super-Bit wins at a matched bit budget - 1.89% against ITQ-128's 4.55% - so the advantage is not
an artifact of giving it 8× the bits. It also wins by **250×** on the metric that turns out to
matter most: distinct fingers mapping to *identical* codes, which no protocol parameter can
reject. ITQ-16's floor of 6,394 collisions in 1.44M pairs (4.4e-3) would be disqualifying on its
own.

Degradation across the three alteration levels is graceful and monotone for all three coders.

### A note on 1.88% vs 1.89%

`07_superbit_eer.py` reports **1.88%** using a sampled impostor set; the table above reports
**1.89%** using all 1,200 × 1,199 pairs. These are the same quantity under two estimators and
differ by 0.01 pp - far inside the confidence interval. Documents elsewhere in this repository
quote 1.88%; that number stands, and **[1.56, 2.20] is the interval to quote with it**.

---

## What to take from this

- The bridge's output contract with the matcher is now *measured*, not assumed: balance, bit
  dependence, and the sub-sample collision probability.
- The bits are dependent enough that the code carries ~10 effective degrees of freedom, but the
  hypergeometric accept model survives that dependence intact.
- Super-Bit is the right choice on accuracy and, more importantly, on the collision floor.
- The binding constraint remains the 16-dimensional embedding, not the code length or the coder.
