# The operating point, measured against the real code distribution

Reproduce with `python 11_operating_point.py`. Raw output: `operating-point.json`, the
Hamming histograms in `operating-point-histograms.npz`, the full 6,848-row parameter sweep
in `operating-point-grid.csv.gz`, and figures in `paper/figures/`.

## Setup

1,200 held-out SOCOFing identities (never seen in training, and disjoint from the images
the quantiser constants were fitted on). Enrolment = `Real`, probe = `Altered-Easy`.
That yields **1,200 genuine** and **2,877,600 impostor** code pairs.

Because the FLPSI accept decision depends only on the total Hamming distance `H` between
two codes, and not on which bits differ, every error rate below is computed **exactly**
from the measured `H` histograms:

```
p(H)      = C(d-H, w) / C(d, w)                     # sub-sample is error-free
accept(H) = P[ Binomial(T, p(H)) >= theta ]
FAR       = sum_H  P_impostor(H) * accept(H)
FRR       = 1 - sum_H P_genuine(H) * accept(H)
```

No resampling, no Monte-Carlo noise. As a check on the arithmetic, `q(8) = C(120,14)/C(128,14)
= 0.38493266…`, matching the published constant to all quoted digits, and evaluating the
same expressions against a uniform-code model reproduces the published per-record FAR of
2.95e-5 exactly.

## Result 1 — the codes are balanced but not uniform

| | uniform model | measured |
|---|---|---|
| impostor mean `H` | 64.0 | **64.0** |
| impostor sd | 5.66 | **20.22** |
| impostor min | ~40 (never observed) | **0** |
| genuine mean `H` | — | 8.3 (sd 5.67, max 34) |

The mean is *exactly* right. The quantiser's balancing step guarantees it, so **no
first-moment check can detect the problem** — a sanity test that only verifies "impostor
codes look like coin flips on average" passes cleanly. The failure lives entirely in the
second moment and the tail: the standard deviation is **3.6× larger** than the model
assumes.

The cause is structural. A 128-bit Super-Bit code is 128 projections of a
**16-dimensional** embedding, so it has roughly 16 real degrees of freedom, not 128. The
LSH bridge is faithful — `E[H]/d = arccos(cos-sim)/π` holds — and *that faithfulness is
the problem*: it transmits the embedding's angular distribution intact, including the
fact that some fingers genuinely look alike.

## Result 2 — the published operating point is off by three orders of magnitude

At the published parameters (`d=128, T=64, w=14, θ=2`, error-free sub-samples):

| quantity | uniform model | measured | ratio |
|---|---|---|---|
| per-record FAR | 2.954e-05 | **4.578e-02** | **1,550×** |
| FRR | — | 9.61e-03 | — |

## Result 3 — a per-record FAR is not a security boundary

A 1:N identification query is compared against every enrolled record, so the rate that
matters is `FAR_query = 1 − (1−FAR)^N`:

| N | 1 | 10 | 100 | 1,000 | 5,000 | 100,000 |
|---|---|---|---|---|---|---|
| `FAR_query` | 0.046 | 0.374 | **0.991** | 1.000 | 1.000 | 1.000 |

FRR stays at 0.96% throughout. **The published parameters describe a verification (1:1)
system, not an identification system**: at N=1 a 4.6% FAR is poor but coherent; by
**N=100** the query-level FAR already exceeds 50%, and at the N=5,000 scale the paper
claimed, an impostor is accepted essentially with certainty.

## Result 4 — retuning (w, t, θ) does not repair it

Sweeping 6,848 combinations of `w ∈ {8…64}`, `t ∈ {0…8}`, `θ ∈ {1…64}` and keeping only
points whose FAR estimate is supported by the histogram (less than half the FAR mass from
bins holding fewer than 10 observed pairs):

| query-FAR target at N=5,000 | best point | FRR |
|---|---|---|
| ≤ 1e-1 | w=56, t=1, θ=40 | **84.8%** |
| ≤ 1e-2 | unreachable | — |
| ≤ 1e-3 | unreachable | — |
| ≤ 1e-4 | unreachable | — |

The only point that even reaches a 10% query FAR rejects 85% of legitimate users. This is
not a tuning problem.

## Result 5 — why: an irreducible collision floor

**8 of the 2,877,600 impostor pairs have Hamming distance exactly 0** — different fingers
producing *identical* 128-bit codes.

```
collision rate p0 = 2.78e-6      (exact Poisson 95% CI [1.20e-6, 5.48e-6])
```

No choice of `w`, `t` or `θ` can reject a pair at `H = 0`: `accept(0) = 1` for every
parameter set, because an error-free sub-sample is guaranteed. So `p0` is a **hard lower
bound on the per-record FAR of any FLPSI parameterisation over these codes**, and it
propagates:

| N | 100 | 1,000 | 5,000 | 10,000 |
|---|---|---|---|---|
| irreducible `FAR_query` | 0.0003 | 0.0028 | **0.0138** | 0.0274 |
| 95% CI | — | [0.001, 0.006] | [0.006, 0.027] | [0.012, 0.053] |

This is exactly why the ladder in Result 4 goes unreachable below 1e-2: at N=5,000 the
floor alone is 1.4%. The near tail is dense too — 1.03e-3 of impostor pairs sit at
`H ≤ 10`, and 2.80e-2 at `H ≤ 25`, which is the *99th percentile of the genuine
distribution*. The genuine and impostor distributions genuinely overlap; the protocol is
being asked to separate distributions that are not separable.

**The bound is a property of the code, not of the protocol.** It can only be moved by
producing better codes — a higher-dimensional or better-trained embedding, or more
independent evidence.

## Result 6 — multi-finger fusion is the lever that works

Each finger is an independent enrolment, so `k`-of-`K` fusion multiplies evidence rather
than trading FAR against FRR along a single curve. Measured over **201 held-out subjects
contributing ≥3 fingers each**, using a Poisson-binomial over the real per-finger accept
probabilities — so correlation between one person's own fingers is carried by the data,
not assumed away:

| parameters | rule | per-record FAR | `FAR_query` @ N=5,000 | FRR |
|---|---|---|---|---|
| w=14, θ=2 | 1-of-3 | 1.31e-01 | 1.000 | 0.000 |
| w=14, θ=2 | 2-of-3 | 8.73e-03 | 1.000 | 0.001 |
| w=14, θ=2 | 3-of-3 | 3.19e-04 | 0.797 | 0.026 |
| w=24, θ=2 | 1-of-3 | 1.95e-02 | 1.000 | 0.002 |
| w=24, θ=2 | 2-of-3 | 2.45e-04 | 0.707 | 0.033 |
| **w=24, θ=2** | **3-of-3** | **1.30e-06** | **0.0065** | **0.281** |

Fusion buys four orders of magnitude on per-record FAR — far more than any single-finger
retuning — and it is the *only* mechanism here that gets below the single-finger collision
floor, because three fingers must collide simultaneously.

But it is not sufficient on its own: the best measured configuration still costs a **28%
FRR**. Honest summary: at N=5,000 with this embedding, there is no configuration of this
protocol that is simultaneously secure and usable.

## What this means for the system

1. **Report the published parameters as a verification (1:1) operating point.** They are
   coherent there and incoherent at 1:N.
2. **Parameters must be fitted to the measured code distribution.** The uniform-code
   assumption is off by 1,550× on FAR here, and it fails silently because the first moment
   is exactly correct.
3. **Publish the collision floor with any FLPSI-over-learned-codes system.** It bounds
   every parameter choice and is cheap to measure: count `H = 0` pairs on a held-out set.
4. **The remedy is more evidence, not a different threshold.** Multi-finger fusion gains
   four orders of magnitude; retuning gains none.
5. **The binding constraint is the embedding.** 16 dimensions cannot keep 5,000 identities
   apart at cryptographic error rates. Raising the embedding dimension is the change that
   would move every number in this document.
