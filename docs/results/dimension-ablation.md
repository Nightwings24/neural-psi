# §7.6 Quantisation-dimension ablation - EER vs Super-Bit code length d

Same open-set protocol as the EER ladder (held-out identities, Real enroll vs Altered-Easy
probe). Super-Bit = ceil(d/16) orthonormalised 16x16 blocks; plain (no whitening); 224 model.

| d (bits) | Super-Bit blocks | storage B/user | EER | genuine H | impostor H | d-prime | bit-balance |
|---|---|---|---|---|---|---|---|
| 64 | 4 | 8 | 2.38% | 4.16 | 32.06 | 3.656 | 50.1% |
| 128 | 8 | 16 | 1.88% | 8.34 | 64.11 | 3.778 | 50.2% |
| 256 | 16 | 32 | 1.89% | 17.05 | 128.26 | 3.814 | 50.0% |

- genuine 1200 pairs, impostor 240000 pairs.
- Accuracy saturates by d=128: 64->128 is a real gain, 128->256 is marginal, while storage
  grows linearly (8/16/32 B/user). Online communication is ~flat in d (TLPSI part dominates,
  see §7.3). The FLPSI backend is hardwired to d=128 (EQ128 circuit + 128-element OPRF key),
  so 128 is also the natively supported point => d=128 is the knee of the trade-off.
