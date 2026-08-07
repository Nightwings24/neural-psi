# Efficiency results — communication (§7.3), latency (§7.4), storage (§7.5)

Measured on the real flash-psi binary (`implementation/flash-psi/src/bin/simulation.rs`, `--sim-type flpsi
--track-io`), op-point **weight=14, t=2, T=64, d=128** (our Super-Bit point). Communication is the
`TrackChannel` total (bytes written + read on the query channel = both directions). The garbled
circuits + garbler labels are pre-shared in setup, so the tracked figure is the **online / per-query**
communication (OT + TLPSI); circuit transfer is one-time, data-independent preprocessing.

Phase timings (this machine: RTX 4050 6 GB; CNN client-side):
- CNN forward: **0.80 ms/image** (GPU) · 13.2 ms/image (CPU), 224×224.
- Quantise (Super-Bit transform, one 128×16 matmul + threshold): **2.2 µs/embedding** (negligible).

## §7.3 Communication vs database size N (single authentication)

| N (db size) | online comm | per-record |
|---|---|---|
| 100    | 0.40 MB  | — |
| 250    | 0.63 MB  | — |
| 500    | 1.00 MB  | — |
| 1,000  | 1.76 MB  | — |
| 2,500  | 4.01 MB  | — |
| 5,000  | 7.77 MB  | — |
| 10,000 | 15.28 MB | — |
| 25,000 | 37.83 MB | — |
| 50,000 | 75.40 MB | ~1.54 KB/record (asymptotic) |

Linear: `comm ≈ 0.25 MB + 1.54 KB·N`. Extrapolates to **~150 MB @ 100 K** and **~1.5 GB @ 1 M** —
matching Bui–Cong's reported **153 MB / 1533 MB**, confirming NeuralPSI inherits FLPSI's communication
profile (the quantiser adds nothing to comm; it only sets the 128-bit width the protocol already uses).

## §7.4 End-to-end latency breakdown (one authentication)

| phase | N = 1,000 | N = 5,000 | scaling |
|---|---|---|---|
| CNN forward (client, GPU) | 0.80 ms | 0.80 ms | O(1) in N |
| Quantise (client) | 0.002 ms | 0.002 ms | O(1) in N |
| **FLPSI online (query-time)** | **187 ms** | **823 ms** | ~0.16 ms/record |
| — total query-time latency | **~188 ms** | **~824 ms** | dominated by FLPSI online |
| FLPSI offline (setup, preprocessing) | 553 ms | 1,946 ms | ~0.38 ms/record + ~240 ms fixed |

- Client-side cost (CNN + quantise) is **< 1 ms** on GPU — the embedding/binarisation bridge is not the
  bottleneck; the server-side FLPSI linear scan is.
- FLPSI **offline** (VOLE-OT setup + garbling + TLPSI construction) is data-independent and can run
  before the query arrives, so it does not count against query-time latency.
- Online scan is **~0.16 ms/record**, consistent with the earlier standalone flash-psi timing.

## §7.5 Storage (per user, server-side)

| system | per-user template | one-time key | 512 users |
|---|---|---|---|
| Blind-Touch (HE/CKKS) | ~1.67 KB (855 KB / 512) | 117 MB Galois key | ~855 KB + 117 MB |
| **NeuralPSI** | **16 B** (128-bit code) | ~0 | **8 KB** |

Per-template: **16 B vs 1.67 KB (~107× smaller)**; plus the **117 MB Galois key is eliminated**. For a
512-user database the totals are ~8 KB vs ~118 MB.

*Caveats:* numbers are a single-host loopback simulation (UnixStream, no real network RTT, no
parallelism); they measure protocol bytes + compute, not wire latency. FPR on random dummy inputs is
≤ ~2e-4 per query (≈ N × per-pair 7e-6), consistent with the design; real-code FAR is in
`realcrypto_results.md`. Reproduce: `for M in 100 500 1000 5000 10000 50000; do ./target/release/simulation
--sim-type flpsi -m $M -w 14 -s 64 -t 2 --track-io --csv; done`.
