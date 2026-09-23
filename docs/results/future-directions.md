# Neural-PSI - future directions to try

Follow-ups grounded in what this session measured (see `improvement-findings.md`).
Each item: the idea, the hypothesis, why our results motivate it, effort, and the honest
expected outcome / risk. Tags: **[cheap]** hours, **[medium]** a day, **[hard]** multi-day or
needs a backend/architecture change. Ordered within each group by value-per-effort.

---

## A. Finish and harden what already works (same-sensor)

### A1. Complete the accuracy ladder for the ortho-thermometer bridge  **[cheap]**
- **Do:** run the new bridge on Altered-Medium and Altered-Hard (we only measured Altered-Easy: 0.87%). Super-Bit's were 4.48% / 7.04%.
- **Why:** the headline (0.87%) is one split; reviewers will want the full ladder + DET curves.
- **Expected:** the ~2× advantage likely holds across splits; confirm with identity-bootstrap CIs.

### A2. Ablate the ortho-thermometer  **[cheap]**
- **Do:** isolate the two ingredients - orthogonalization vs magnitude thresholds - vs Super-Bit and rand-thermometer (we have most of these numbers; make it a clean 2×2 ablation with CIs).
- **Why:** the gain came from *combining* both (0.79% ortho vs 1.14% rand vs 1.89% Super-Bit); a clean ablation makes the mechanism defensible.

### A3. Optimize the thermometer design  **[medium]**
- **Do:** sweep K×Bp (projections × thresholds), threshold placement (quantile vs learned vs Gray-coded), and total bit budget; check FLPSI `q(H)` still holds for each.
- **Hypothesis:** better threshold placement narrows impostor σ further (currently 14.6 vs uniform 5.66), lifting both EER and the FAR frontier.
- **Risk:** learned thresholds weaken the "public, training-free" property - analyze that tension.

### A4. Larger-N real-crypto validation  **[medium]**
- **Do:** push the real flash-psi binary well past N=100 (we validated at N=100; the closed-form claims rest on the surrogate at 2.88M pairs).
- **Why:** closes the "your headline never touched the real binary at scale" reviewer objection.

---

## B. The remaining losses (comms, at-rest, 1:N)

### B1. Communication reduction  **[medium]**
- **Do:** sweep sub-sample count T ∈ {16,32,64} through `simulation --track-io`; plot comms vs FAR/FRR; explore OPRF batching.
- **Why:** comms (7.77 MB @ N=5000, ~10× Blind-Touch) is the one clean *loss* left; retuning (w,t,θ) does NOT touch FAR (measured), but T might cut comms.
- **Risk:** modest gain; too-small T worsens FAR.

### B2. Template-at-rest: implement + measure the per-enrolment XOR/OTP mask  **[medium]**
- **Do:** store `code ⊕ r_u` with client-side `r_u`; measure that accuracy is unchanged (XOR preserves Hamming) and that it blocks DB-dump replay.
- **Why:** the security analysis showed this recovers unlinkability + renewability against a DB-dump adversary (one-time-pad argument) at zero accuracy cost - a real, cheap ISO/IEC 24745 win. Be honest it does NOT give irreversibility (linear map still invertible under full compromise) and it makes the system two-factor (finger + device secret).

### B3. Fusion refinements  **[cheap-medium]**
- **Do:** K>3 quorums, quality-aware finger selection, weighted (per-finger-reliability) fusion; report FRR/FAR/comms tradeoffs with CIs.
- **Why:** ortho-thermo 3-of-3 already hits 0.9% FRR at a secure point; weighting/selection may push further or reduce the 3-finger enrolment cost.

### B4. Malicious-security upgrade (as a costed plan, not a claim)  **[hard]**
- **Do:** spec authenticated-garbling mOPRF + malicious VOLE + input-consistency; give concrete overhead.
- **Honest caveat (measured reasoning):** the three concrete attacks are *input-substitution*, which malicious MPC does **not** fix - so frame this as closing protocol-*deviation* only, never as fixing impersonation.

---

## C. Cross-sensor / real-world generalization (the hard wall)

We showed contact↔contactless is at chance even when trained (float EER ~50%), while same-sensor works (14-19%). These target that wall - high value, higher risk.

### C1. Replicate the bridge + floor findings on a SECOND same-sensor real dataset  **[medium]**
- **Do:** FVC2002/2004 or NIST SD302 (SD302 has enough fingers to measure the ~1e-6 floor; FVC does not - use it for EER only).
- **Why:** cheapest path to *generality* - if the metric-mismatch + collision-floor + bridge-win reproduce on a different real corpus, the contribution stops being "a SOCOFing result." Higher ROI than cracking cross-sensor.
- **Expected:** the *phenomena* likely replicate even if absolute EER is worse (real data ≈ 14-19% same-sensor).

### C2. Stronger / wider embedding + longer code, co-designed  **[hard]**
- **Do:** a bigger backbone (e.g. DeepPrint-style, 512-D) AND a longer code - but this needs a FLPSI backend change (see D1), since a fixed 128-bit code can't carry more DOF (measured).
- **Hypothesis:** the float ceiling improves with dimension (measured: 0.33→0.14%); if the code can grow to match, cross-sensor *might* clear chance.
- **Risk:** cross-sensor float EER was at chance even at 16-D trained; a wider net may still not bridge the domain gap without alignment (C3).

### C3. Domain adaptation for contact↔contactless  **[hard]**
- **Do:** minutiae/ridge alignment or preprocessing (the dataset's own paper, Lin & Kumar 2018, uses alignment); domain-adversarial training; or a CycleGAN-style contact↔contactless image translation before embedding; hard-negative / contrastive losses.
- **Why:** the gap is representational; the literature closes it with explicit alignment, which we did not attempt.
- **Risk:** substantial ML effort; drifts toward an ML-primary paper (against SPACE scope) - keep the crypto contribution central.

### C4. More data / epochs on PolyU  **[cheap]**
- **Do:** train longer (>60 epochs) with all 336 fingers and augmentation; check if same-sensor improves and whether cross-sensor ever clears chance.
- **Why:** our run was 60 epochs / 252 fingers, pair_acc still climbing - a cheap check of whether the wall is capacity or just under-training. **Expected:** same-sensor improves; cross-sensor probably stays near chance.

---

## D. Protocol / systems

### D1. Code-length co-design (decouple from the 128-bit backend)  **[hard]**
- **Do:** extend flash-psi to support d ≠ 128 (EQ circuit width + OPRF key length), then test higher-D embedding + longer code jointly.
- **Why:** the measured bottleneck is that 128 bits can't carry more DOF; this is the only way to test whether embedding width helps once the code can grow.

### D2. Multi-shard scaling experiment  **[medium]**
- **Do:** the two-shard scaling the paper alludes to - measure latency/comms vs N across shards on the real binary.

### D3. Same-hardware Blind-Touch head-to-head  **[medium]**
- **Do:** use the in-repo Docker baseline for a controlled comms + total (offline+online) wall-clock comparison (the paper currently quotes the original cluster numbers; the unverifiable 1,334 ms figure should be dropped).

---

## E. Rigor / reproducibility (for submission)

- **E1.** Identity-bootstrap CIs + DET curves on *all* new numbers (fusion FRR, operating points), not just EER.  **[cheap]**
- **E2.** Negative controls: label-shuffle and untrained-CNN baselines (confirm the measured floor is a learned-embedding effect, not a counting artifact).  **[cheap]**
- **E3.** Finger-type stratification of the H=0 collisions (rebut the "right-thumb artifact" objection).  **[cheap]**
- **E4.** `provenance()` stamping (commit + seed + dataset SHA) on every artifact + a `make reproduce` target + a RESULTS-MANIFEST mapping each number to its command.  **[medium]**

---

## Recommended next three (highest value-per-effort)

1. **A1 + A2** - finish the accuracy ladder + ablation for the new bridge (cheap, makes the headline submission-ready).
2. **C1** - replicate on a second real same-sensor dataset (SD302/FVC): the cheapest route to *generality*, directly defusing the "SOCOFing-only" rejection.
3. **B2** - implement + measure the XOR/OTP at-rest mask: a real ISO/IEC 24745 win at near-zero cost, strengthening the security story.

Everything cross-sensor (C2/C3) is high-risk; pursue only if the same-sensor story (A/B/C1) is locked first.
