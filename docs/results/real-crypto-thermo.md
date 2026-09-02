# Real-crypto head-to-head — Super-Bit vs L2-thermometer bridge

Both bridges' codes for the SAME 100 held-out fingers pushed through the actual flash-psi
binary (masked-OPRF+GC+VOLE+Shamir). Op-point w=14, t=2, T=64, d=128.
100 genuine + 9900 impostor decisions per bridge.

| bridge | real TAR | real FAR | pred FAR | gen H | imp H | ms/run |
|---|---|---|---|---|---|---|
| Super-Bit | 100.0% | 4.42e-02 | 4.45e-02 | 8.6 | 64.3 | 66.2 |
| L2-thermo | 100.0% | 1.05e-01 | 1.05e-01 | 7.0 | 48.5 | 67.4 |

Confident-regime correctness (real must agree with the near-certain model):

- **Super-Bit**: confident-REJECT 8519/8525, confident-ACCEPT 94/94.
- **L2-thermo**: confident-REJECT 6516/6530, confident-ACCEPT 100/100.
