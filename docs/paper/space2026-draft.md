# (working title)
**Quantization-Aware Binarisation for Accurate and Communication-Efficient Fuzzy-PSI Fingerprint Authentication**


---

## Abstract

Privacy-preserving fingerprint authentication increasingly pairs a neural feature
extractor with a cryptographic matcher, yet the interface between them is usually an
afterthought: the binariser that turns a real-valued embedding into the fixed-length code
the protocol actually compares. We study this interface in Neural-PSI, a system that
authenticates a fingerprint through fuzzy-labelled private set intersection over 128-bit
codes, in place of the homomorphic-encryption matcher used by prior neural approaches. Our
central observation is that this binariser is at once the accuracy bottleneck and the
communication driver, because both are governed by one quantity — the impostor Hamming
distribution of the code — through the protocol's subsample-based accept rule. A single,
better code therefore improves both, and the two goals that usually trade off against each
other here move together. We realise this with a quantization-aware code, learned from the
extractor's own features to be balanced and decorrelated, which pushes the code toward the
uniform distribution the protocol implicitly assumes. On SOCOFing the learned code lowers
the equal-error rate from 0.87% to 0.17% — roughly fivefold, with non-overlapping
confidence intervals and holding across the full altered-difficulty ladder — and cuts online
communication by 45% at a matched operating point, on the same 128-bit backend and measured
end-to-end against the real protocol rather than a cost model. Across three datasets we find
the communication gain to be universal, while the accuracy gain scales with the quantization
gap between the float embedding and its code; we characterise this dependence and its
practical consequences for deployment.

---

## Notes on the abstract (for our iteration, not for submission)

- **What it claims, and how honestly:** the ~5× and −45% are stated as *SOCOFing* results
  (our high-fidelity corpus), and the last sentence pre-empts the "single-dataset" objection
  by promoting the honest cross-corpus finding (universal comms gain, gap-dependent accuracy)
  into a contribution rather than hiding it.
- **What it deliberately omits** (kept for the body, not the abstract): the metric-aware
  ortho-thermometer bridge (round 1), the collision-floor analysis, the fusion tradeoff, and
  the security/template-protection discussion. If a reviewer signal says the security angle
  must lead (SPACE is a crypto venue), we can swap one sentence to foreground it.
- **Open decisions:** (1) name Blind-Touch explicitly vs. "prior neural approaches"; (2)
  whether to cite the exact FLPSI construction (Bui–Cong) in the abstract; (3) whether to add
  one clause on the real-crypto validation being through the actual Rust binary.

### Alternative titles
1. *Co-Designing the Code: Quantization-Aware Binarisation for Accurate and
   Communication-Efficient Fuzzy-PSI Fingerprint Authentication*
2. *The Binariser is the Bottleneck: Joint Accuracy–Communication Gains in Fuzzy-PSI
   Biometric Authentication*
3. *One Code, Two Wins: Quantization-Aware Fuzzy-PSI for Fingerprint Authentication*
