# Neural-PSI → SPACE 2026: Research Plan

**Status:** synthesis of a four-wave multi-agent review (red-team, related-work, novelty,
methodology, security, experimental-design, + a targeted prior-art verification pass).
**Repo state:** git `f855367`. **Venue:** SPACE 2026 (Security, Privacy and Applied
Cryptographic Engineering), ≤20 pp LNCS, double-blind, Bangalore, Dec 16–19 2026.

### Provenance convention (binds every empirical claim in this file)
- `[MEASURED: <script/artifact> @ <commit>]` — produced by code in this repo; re-runnable.
- `[PUBLISHED: <author year>, <venue>, <loc>, <URL>]` — read this session from the primary source.
- `[PROPOSED]` — hypothesised, **not yet measured**; the exact experiment is given.

Anything untagged is analysis/opinion, not a result.

---

## 1. The one defensible contribution

> **When the bit-strings fed to a fuzzy-labelled-PSI matcher are produced by an LSH bridge
> over a *learned* biometric embedding rather than the uniform-code idealisation the
> protocol's error analysis assumes, the standard parameter selection is wrong by three
> orders of magnitude on false-accept rate, the failure is invisible to any first-moment
> sanity check (mean impostor distance is exactly d/2 by construction while its standard
> deviation is 3.6× the assumed value), and an irreducible ~10⁻⁶ same-code collision floor
> lower-bounds the false-accept rate of *every* parameterisation — a parameter-selection
> analysis for threshold-PSI biometric matching computed from the measured code distribution
> and validated against the real cryptographic protocol.**

**SPACE topic it lands in** (quoted from `space2026.md`):
> "Design and analysis of applied cryptographic primitives and protocols"

secondarily
> "Cryptography in the wild"

**Why this clears SPACE's scope bar** (which explicitly discourages ML-primary and
purely-theoretical work): the object of study is a *cryptographic protocol's
parameter-selection analysis*. The CNN is a swappable black-box code source — the result is
argued to hold for any LSH-over-learned-embedding. It is not pure theory: the headline is
computed exactly over the measured held-out impostor pairs (1.44×10⁶ canonical
probe×gallery events; 2.88×10⁶ if enrol-vs-enrol pairs are included — see §6 MF-1) and
cross-checked against the real masked-OPRF + garbled-circuit + VOLE + Shamir binary. The system (16-byte templates, no
rotation key) is demoted from "the contribution" to "the artifact we measured on."

This framing is now bounded by *verified* prior art (see §2 and §4):
- The uniform-code assumption we refute is **explicit in the primary construction**:
  `[PUBLISHED: Bui & Cong 2025, CANS/ePrint 2025/1470, §5.3 p.18, https://eprint.iacr.org/2025/1470]`
  — *"Suppose x and y are sampled uniformly at random, then the number of bits that differ
  follows the binomial distribution … Pr[HW(x⊕y)=v] = C(d,v)/2^d,"* feeding their headline
  FPR < 7×10⁻⁶.
- We must **not** claim to be first to observe real-biometric fuzzy-PSI false matches: Uzun
  et al. already report a real-biometric false-match/FRR tradeoff
  `[PUBLISHED: Uzun et al. 2021, USENIX Security, Fig. 7 p.921, https://www.usenix.org/system/files/sec21-uzun.pdf]`.
  Their analysis is *empirical* (no uniform/Binomial model), so it does not pre-empt the
  "silent first-moment failure" or the collision-floor bound — those remain ours.

---

## 2. Competitive landscape

All competitor numbers were read this session from the primary source (verbatim quote +
location on file). "n/r" = not reported / not extracted. This-work rows are `[MEASURED]`.
**Every cross-row comparison is subject to the caveats below the table — most are NOT
apples-to-apples.**

| System | Accuracy | Template/user | Key material | Comms/query | Threat model | Scale note |
|---|---|---|---|---|---|---|
| **This work** (Super-Bit-128 + FLPSI) | EER **1.88%** [1.56, 2.20] SOCOFing/Altered-Easy; float ceiling **0.33%** ¹ | **16 B** ² | **none** ² | **7.77 MB** @ N=5,000 ² | semi-honest ³ | per-record FAR **4.58×10⁻²**; 1:N FAR→1 by N≈1,000 ⁴ |
| This work — real-crypto check | TAR 100%, **FAR 4.43×10⁻²** (pred 4.45×10⁻²), N=100 ⁵ | 16 B | none | n/r | semi-honest, real binary | validates the accept model, not scale |
| **Blind-Touch** (baseline) | F1 **93.6%** PolyU (abstract; peak 93.8% @θ=0.2, Table 4); F1 98.2% / EER **0.7%** SOKOTO ⁶ | 1.67 KB (derived) ⁷ | **Galois 117 MB** + relin 4.5 MB ⁶ | input **0.8 MB** ⁶ | semi-honest / curious server ⁶ | "5,000 in ~0.65 s" = 650 ms, **3-cluster** ⁶ |
| Blind-Match | n/r (ACM 403; header only) | n/r | n/r | n/r | n/r | HE 1:N (scope from title) ⁸ |
| Uzun et al. FLPSI | real-biometric FRR/false-match tradeoff, N up to 1M ⁹ | n/r | n/r | 528 MB @100K, 2124 MB @1M ¹⁰ | semi-honest | face; FaceNet embeddings |
| Bui–Cong FLPSI | FPR < 7×10⁻⁶ (**uniform-code model**) ¹¹ | d/8 B | PRF key | **153 MB @100K, 1533 MB @1M** ¹² | semi-honest ¹³ | the construction we instantiate |
| ITQ-128 (our baseline coder) | EER **4.55%** [3.91, 5.31]; **517** H=0 / ~1.44M ¹ | 16 B | n/a | n/a | n/a | matched-bit-budget control |
| ITQ-16 | EER **6.13%** [5.54, 6.80]; 6,394 H=0 / ~1.44M ¹ | 2 B | n/a | n/a | n/a | bit-budget mismatch |

**Provenance keys.**
1. EER 1.88% / ITQ EERs / H=0 counts `[MEASURED: 12_code_analysis.py @ f855367, code-analysis.json]`; float ceiling 0.33% `[MEASURED: 07_superbit_eer.py @ f855367, docs/results/accuracy-eer.md]` (code-analysis.json has no float entry).
2. `[MEASURED: docs/results/efficiency.md @ f855367]` (loopback simulation; see caveat C2).
3. `[MEASURED: README/paper scope @ f855367]`.
4. `[MEASURED: 11_operating_point.py @ f855367]` (docs/results/operating-point.md Results 2–3).
5. `[MEASURED: 09_realcrypto_validate.py @ f855367]` (docs/results/real-crypto-validation.md).
6. `[PUBLISHED: Choi, Woo & Kim 2024, AAAI, Tables 2/4/5/7/8 & Abstract, arXiv 2312.11575]`.
7. Derived (856 KB ciphertext ÷ 512 slots); **not verbatim** in the paper — label as derived.
8. `[PUBLISHED: Choi et al. 2024, CIKM, arXiv 2408.06167]` — header verified, numbers not extracted.
9. `[PUBLISHED: Uzun et al. 2021, USENIX Security, Fig. 7 p.921]`.
10. `[PUBLISHED: Bui & Cong 2025, ePrint 2025/1470, Table 3 p.23]` (the [UCK+21] column).
11. `[PUBLISHED: Bui & Cong 2025, ePrint 2025/1470, §5.3 p.18 (uniform quote) + §6.2 p.22 (FPR<7×10⁻⁶)]`.
12. `[PUBLISHED: Bui & Cong 2025, ePrint 2025/1470, Table 3 p.23]` ("ours" column) — confirmed verbatim.
13. FLPSI family; semi-honest.

### Comparability caveats (read before any "we beat X")
- **C1 — accuracy is NOT comparable.** Our EER is SOCOFing/Altered-Easy (synthetic
  alterations of one capture); Blind-Touch is F1@threshold on PolyU/SOKOTO. Different
  datasets, different metric definitions. Do **not** juxtapose 0.33%/1.88% against 0.7%.
- **C2 — latency is NOT comparable.** Blind-Touch's 650 ms is a 3-cluster distributed
  figure; our ~824 ms is single-host loopback (no network RTT, no parallelism), online-only
  (excludes ~1,946 ms offline @ N=5,000). **The README's "1,334 ms single-server"
  Blind-Touch figure could not be verified and must be removed** (§6, MF-2).
- **C3 — comms is a LOSS.** 7.77 MB vs 0.8 MB (~10×). Reported honestly; keep it that way.
- **C4 — the only fully fair head-to-head is Super-Bit vs ITQ** (our own pairs, matched
  128-bit budget, same code): 1.88% vs 4.55% EER, 2 vs 517 collisions.
- **C5 — defensible clean wins vs Blind-Touch:** key-material elimination (117 MB → 0) and
  template size (16 B). Everything else is a loss, a tie, or not comparable.

---

## 3. Prioritised technical roadmap (research-value-per-effort, highest first)

### R1 — Embedding-dimension sweep *(the spine of the paper)*
- **Hypothesis.** The wide impostor spread (sd 20.11) and the H=0 collision floor are
  consequences of the code carrying only ~10 effective degrees of freedom
  `[MEASURED: 12_code_analysis.py @ f855367 — participation ratio 9.84 of 128]` (far fewer
  than the nominal 16 input dimensions), not of the coder or protocol. Raising `EMB_DIM`
  narrows both.
- **Expected effect [PROPOSED].** As D grows (16→32→64→128→256→512), impostor sd falls
  toward the uniform 5.66 and p₀ falls toward 0. **Direction only** — no magnitude is
  measured; do not state any sd(D)/p₀(D) number until R1 runs.
- **Exact experiment.** Add `--emb-dim` to `src/train_gpu_224.py` (plumb into
  `SiameseModel(emb_dim=…)`); add `--in-dim` to `07_superbit_eer.py`, `11_operating_point.py`,
  `12_code_analysis.py` (set `QuantizerConfig(in_dim=D, code_len=128)`). code_len stays 128
  (the FLPSI backend is hardwired to EQ128 + 128-element OPRF key
  `[MEASURED: crypto/README.md, dimension-ablation.md @ f855367]`). Retrain per D on the
  RTX-4050-class GPU used originally; re-emit impostor sd, p₀ (with Poisson CI), and EER.
- **Metric that decides it.** sd(D) and p₀(D). Earns the generality claim iff both fall
  toward the uniform model at a reachable D; **refutes** it only if the floor stays
  macroscopic at 512-D — which is itself a *stronger* publishable negative result.
- **Effort.** Medium (5 GPU retrains + CPU re-eval; reuses all infra). **Risk:** the
  code-length ablation `[MEASURED: dimension-ablation.md]` shows accuracy saturates by
  d=128, but that varied *code length at fixed 16-D embedding* — a different axis; conflating
  the two would be an overclaim.
- **Loss→win?** **YES — the single item that can convert the collision-floor and 1,550× FAR
  losses into a win, and the one the generality claim stands on.** Backend/storage unaffected.

### R2 — Cross-sensor replication of the floor on a second dataset *(co-spine)*
- **Hypothesis.** The floor is a property of learned-code FLPSI, not a SOCOFing thumb-quality
  artifact (the repo's own tier-2 note traces near-tail collisions to confusable right-thumb
  prints `[MEASURED: operating-point-and-fusion.md @ f855367]`).
- **Expected effect [PROPOSED].** A non-trivial floor reappears on genuinely different
  capture conditions.
- **Exact experiment.** NIST SD302
  `[PUBLISHED: NIST, https://www.nist.gov/itl/iad/image-group/nist-special-database-302 — free after request]`;
  new extractor `01c_extract_sd302.py` → `train_gpu_224.py` → `11`/`12`. Cross-sensor variant:
  enrol on one sensor, probe on another. **Feasibility (F1):** a ~10⁻⁶ floor needs ≳1,000
  distinct fingers; SD302 qualifies, PolyU/FVC do not (use those for EER only).
- **Effort.** Large (access lead time + extract + retrain). **Risk:** floor absent → the
  SOCOFing floor is a dataset artifact and the "irreducible bound" framing must be retracted.
  This is the experiment that most changes the paper's standing.

### R3 — Rigorous multi-finger fusion *(the strongest positive result; make it bulletproof)*
- **Current evidence.** 3-of-3 (w=24) reaches per-record FAR **1.30×10⁻⁶** and query-FAR
  6.5×10⁻³ @ N=5,000, via a Poisson-binomial over *real* per-finger accept probabilities on
  201 held-out subjects `[MEASURED: 11_operating_point.py @ f855367]` — four orders of
  magnitude below single-finger, and the only lever that crosses the floor.
- **The rigorous gap.** Independence is assumed in the closed form but only spot-checked.
  Measure the *joint* cross-finger collision rate directly (all-K-subset sub-match counting
  through the real `flpsi_match.SubSampler`), sweep K∈{2..5}, k∈{1..K}, w∈{14,24,32}; confirm
  the chosen point through the real Rust `fingerprint` binary as `09` did for single-finger.
- **Report honestly.** The best config still costs **28% FRR**
  `[MEASURED: operating-point.md @ f855367]` — so fusion's win is **conditional on R1** for a
  point that is simultaneously secure *and* usable.
- **Effort.** Low–medium (no GPU, no retrain). **Loss→win?** PARTIAL alone; **R1 × R3
  together** is the plausible route to secure-and-usable, and the paper's strongest positive.

### R4 — Metric-fuzzy-PSI-over-raw-embedding: analysis, not (yet) a build
- **Hypothesis (reviewer's).** Matching the raw 16-D cosine embedding via structure-aware
  fuzzy PSI `[PUBLISHED: Gao et al. 2024, ASIACRYPT, ePrint 2024/1462]` would avoid both the
  +1.55 pp quantisation loss and the code-collision floor.
- **Caution.** Metric PSI removes *quantisation-induced* collisions but not the underlying
  angular overlap — 2.80×10⁻² of impostor pairs already sit within the genuine 99th
  percentile `[MEASURED: operating-point.md @ f855367]`. It likely **relocates** the floor
  (hard collision → soft threshold error), not removes it.
- **Exact experiment (cheap, no crypto).** On cached float embeddings
  (`src/data/dim_ablation_emb_224.npz`), compute impostor FAR at the cosine threshold giving
  the FLPSI-matched genuine FRR. If float-domain FAR ≈ code-domain FAR, metric PSI buys
  accuracy but **not** security — and the paper says so. A fair comparison must also cost the
  metric-PSI protocol's comms/latency; do not assert it is cheaper.
- **Effort.** Low for the bound; high/out-of-scope for a metric-PSI backend. **Loss→win?**
  Speculative; real accuracy headroom (+1.55 pp bounded), uncertain security headroom.

### R5 — Learned coder: run to *defend* Super-Bit, not to replace it
- **Measured priors against.** Super-Bit-128 (1.88%) already beats ITQ-128 (4.55%) at matched
  budget with 250× fewer collisions; whitening *hurts* on the 224 model (1.88%→2.98%);
  ArcFace collapsed in 16-D `[MEASURED: accuracy-eer.md, code-analysis.md @ f855367]`.
- **Tension.** A learned coder weakens the "public, training-free" property that keeps the
  coder outside the crypto TCB, and adds attack surface without adding privacy (the sign map
  is already non-one-way). **Effort.** Medium. **Loss→win?** Unlikely; include to close the
  reviewer question with evidence. One datapoint at R1's best D.

### R6 — Protocol comms sweep (the only legit lever on the comms loss)
- **Do not** frame retuning as a FAR fix — a 6,848-point (w,t,θ) sweep already shows
  query-FAR ≤10⁻² is unreachable at N=5,000 `[MEASURED: operating-point.md Result 4 @ f855367]`.
- **Legit item:** sweep T∈{16,32,64} via
  `./simulation --sim-type flpsi -m … -s <T> --track-io --csv`
  `[MEASURED path: crypto/README.md @ f855367]`, plot comms(T) vs FAR(T). Incremental; may
  cut the 7.77 MB, will not touch the FAR story.

**Rejected (would contradict a measured result or overclaim):** retune-to-fix-FAR;
whitening for accuracy; ArcFace at 16-D; d=256 for accuracy; any "bridge is one-way" claim.

---

## 4. Reviewer-risk section (5 most likely rejections + how the plan defuses each)

1. **"You found that 16-D is too small, not a limitation of FLPSI-over-learned-codes"**
   *(the kill shot — the generality claim rests on n=1 embedding, n=1 dataset).*
   → **Defused only by R1 + R2.** The dimension sweep either earns the generality claim
   (floor persists across D) or honestly reframes it ("16-D-class embeddings are structurally
   unsafe for FLPSI"). Until R1 exists, the generality claim must be softened to "for this
   embedding class." This is the highest-priority work item in the plan.

2. **"The floor is a SOCOFing thumb-quality artifact"** (the repo's own docs say so).
   → **Defused by R2** (SD302 replication) + a finger-type stratification of the collisions
   (nice-to-have N3): show the floor is not thumb-dominated, or say plainly that it is.

3. **"The system doesn't work and loses on comms/at-rest — why is this a paper?"**
   → Reframe: the contribution is the *measurement/analysis* (§1), not the system. The system
   is the vehicle. Wins (16 B, no key) are real `[MEASURED]`; losses (comms, 1:N FAR,
   at-rest) are quantified, not hidden. Pair with R3 for the one positive operating regime.

4. **"Your semi-honest theorem proves the wrong thing"** (client is untrusted in auth).
   → The three concrete attacks are **input-substitution** — a well-formed chosen `x` — which
   **no malicious-MPC upgrade removes** (they are realisable in the ideal world). State this;
   frame malicious security as closing protocol-*deviation* (threat A) only, and mitigate
   input attacks with rate-limiting + sensor attestation + fusion. State the corollary that a
   negligible ε coexists with total on-accept record recovery (the proven guarantee ≠ the
   deployment guarantee).

5. **"Novelty vs Bui–Cong / Uzun."**
   → Bounded by verified reading: the uniform-assumption critique is genuine (Bui–Cong §6.1
   p.18); the 1:N FAR *statistical* composition and the silent-first-moment / collision-floor
   findings are ours; Uzun's empirical false-match result is cited and not overclaimed. **The
   paper's current "N-record composition is an undischarged gap" claim is FALSE and must be
   corrected** (§6, MF-3).

---

## 5. Experiment & reproducibility plan

### Must-have for credibility
| # | Question | Dataset (verified) | Command | Metric | Effort |
|---|---|---|---|---|---|
| R1 | Floor & sd vs embedding dim | SOCOFing (in hand) | `train_gpu_224.py --emb-dim D` → `11`/`12 --in-dim D` | sd(D), p₀(D)+Poisson CI, EER | M |
| R2 | Floor replicates cross-sensor | NIST SD302 (free/request) | `01c_extract_sd302.py` → train → `11`/`12` | p₀, sd, EER | L |
| M3 | Cross-sensor EER | PolyU contact↔contactless (free/agreement) | `01d_extract_polyu.py` → `07`/`11` | EER [CI] | L |
| M4 | Same-hardware Blind-Touch head-to-head | SOCOFing | `baseline-blind-touch/` Docker + new `bench_headtohead.py` | total wall-clock + bytes @ N∈{100,1k,5k} | M |
| M5 | Real-crypto agreement at N≫100 | SOCOFing | `09_realcrypto_validate.py --batch B --queries Q` | real vs predicted FAR/TAR | M |

### Nice-to-have
Label-shuffle & untrained-CNN negative controls (N1/N2); finger-type stratification of
collisions (N3); quantiser-stage ablation with CIs (N4); seed sensitivity (N5); k-fold over
all 6,000 fingers for tighter CIs (N6); FVC EER benchmark (N7).

### Statistical rigor & reproducibility (build on what exists)
- Identity-level bootstrap CIs already exist (`12_code_analysis.py:eer_ci`); the operating
  point is exact histograms, no Monte-Carlo (`11_operating_point.py`). Keep both.
- Add a `provenance()` stamp (git commit + seed + dataset SHA-256 + lib versions) to every
  artifact; a `make reproduce` target for the canonical order
  (`01b→train→07→11→12→14→09`); a `RESULTS-MANIFEST.md` mapping every published number →
  {value, CI, script, commit, seed, dataset}.
- Feasibility flags: floor needs ≳1,000 fingers (SOCOFing/SD302 only); SD302/PolyU gate on
  access requests (start now); real-crypto scaling is CPU-bound; the HE baseline's accuracy
  train is the long pole for M4 (a low-epoch model suffices for *timing*).

---

## 6. Must-fix integrity items (before submission; each would be a Wave-4 defect)

- **MF-1 — collision-floor number, now RESOLVED by in-session measurement.** The "2 (of
  1.44M)" and "8 (of 2.88M)" figures are **both correct** — they use different impostor sets.
  `[MEASURED: in-session over src/data/sb_codes_224.npz @ f855367]`:
  - Canonical (probe×gallery, off-diagonal = the operational 1:N event): **2 / 1,438,800 =
    p₀ 1.39×10⁻⁶, exact Poisson 95% CI [1.68×10⁻⁷, 5.02×10⁻⁶]** (impostor mean 64.05, sd 20.11).
  - `operating-point.md`'s 8 / 2,877,600 = 2.78×10⁻⁶ *adds the gallery×gallery block*
    (enrol-vs-enrol pairs, which are not queries).
  → **Adopt the canonical probe×gallery definition everywhere; report p₀ = 1.39×10⁻⁶ with its
  CI as the 1:N floor; footnote the 2.78×10⁻⁶ as the definition that includes enrol pairs.**
- **MF-2 — remove the unverifiable Blind-Touch "1,334 ms single-server" latency** (README:39);
  the paper's own 650 ms (3-cluster) is the sourced figure. Do not present an "824 < 1,334" win.
- **MF-3 — correct the REFUTED composition claim.** `06-security.tex` §6.1 states the
  N-record composition is "a gap a citation cannot discharge"; Bui–Cong Theorem 6 (+ Thm 4)
  discharge exactly the m-record, shared-PRF-key, shared-receiver-polynomial case with
  leakage ≈ `L_C` `[PUBLISHED: ePrint 2025/1470, Fig. 10 + Thm 6]`. Re-scope crypto novelty
  to: the explicit functionality/leakage write-out (rigor), the **multi-session** k-reuse
  accounting (a real [PROOF-SKETCH] gap — where attack (c)'s accumulating leakage lives), and
  the empirical uniform-assumption refutation.
- **MF-4 — disambiguate the Blind-Touch PolyU F1 (do NOT "fix a typo").** The paper reports
  **93.6%** in the abstract/body and **93.8%** as the per-threshold peak (Table 4, θ=0.2)
  `[PUBLISHED: Choi, Woo & Kim 2024, AAAI, arXiv 2312.11575, Abstract + Table 4]` — both are
  genuine, different quantities; state which one we mean wherever we cite it. There is **no
  PolyU F1 figure in the README** to correct (an earlier draft of this plan wrongly asserted
  one). Separately, label the "1.67 KB" per-user template as *derived* (856 KB ciphertext ÷
  512 slots); the 856 KB ciphertext size itself is verbatim `[PUBLISHED: AAAI, Table 2]`.
- **MF-5 — strengthen (don't just concede) the XOR/ISO-24745 story.** Against a DB-dump-only
  adversary (D₁), XOR with client-side `r_u` is a one-time pad → hides the stored code, blocks
  replay, gives unlinkability + renewability [PROOF-SKETCH, OTP]; against full compromise
  (D₂) it buys nothing and the linear quantiser is provably invertible. Present a per-adversary
  table; note XOR makes it two-factor (finger + device secret). True irreversibility needs a
  fuzzy extractor the paper does not implement.
- **MF-6 — quarantine superseded rosy fusion numbers** (`operating-point-and-fusion.md` still
  shows 3-of-3 @ ~5% FRR vs the authoritative 28%); reconcile README "not vendored" with the
  in-tree `crypto/flash-psi/`.

---

## Achievable vs aspirational (stated plainly)

- **Achievable now (in-hand data/binaries):** the canonical floor number (done, MF-1); R4's
  float-domain FAR bound; R6's comms sweep; R3's joint-collision analysis; all MF corrections;
  M5 at moderate N.
- **Achievable with GPU time (no new data):** R1 (the spine). This is the single highest-value
  item and gates the paper's central claim.
- **Aspirational / access-gated:** R2 and M3 (SD302, PolyU) — start the data requests now;
  they carry lead time and their outcomes are genuinely uncertain.
- **Rejected as overclaiming:** anything that presents the system as deployable at 1:N, the
  1,334 ms latency win, malicious security as a fix for impersonation, XOR as irreversibility,
  or the composition as an undischarged proof gap.
