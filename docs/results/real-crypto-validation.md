# Real-crypto validation — Super-Bit codes through the actual flash-psi binary

Our held-out 224-model Super-Bit codes pushed through the REAL masked-OPRF + garbled-circuit
+ VOLE + Shamir protocol (`implementation/flash-psi/src/bin/fingerprint.rs`), compared against the
closed-form sub-sampling model we used in Tier-1/Tier-2. Agreement => the surrogate is faithful
and the bridge is real-crypto-correct.

- Operating point: weight=14, t=2, T=64; plain Super-Bit (no whitening), d=128.
- Database B=100 held-out enrollment (Real) codes; 100 probe (Altered) queries.
- Decisions: 100 genuine + 9900 impostor, through real crypto.
- Per-run latency: mean 56.5 ms (full setup+online, m=100).

| metric | REAL (crypto) | PREDICTED (q-model) |
|---|---|---|
| TAR | 100.00% | 99.52% |
| FAR | 4.43e-02 | 4.45e-02 |
| genuine Hamming (mean/max) | 8.59 / 24 | — |
| impostor Hamming (mean/min) | 64.35 / 4 | — |

Accept-rate by Hamming bucket (real vs closed-form):

| H bucket | n | real | predicted |
|---|---|---|---|
| [0,10) | 64 | 100.00% | 100.00% |
| [10,20) | 130 | 98.46% | 99.64% |
| [20,25) | 148 | 86.49% | 88.86% |
| [25,30) | 229 | 58.52% | 55.45% |
| [30,35) | 267 | 20.97% | 21.31% |
| [35,40) | 371 | 6.47% | 6.08% |
| [40,50) | 1168 | 0.43% | 0.69% |
| [50,129) | 7623 | 0.00% | 0.00% |
