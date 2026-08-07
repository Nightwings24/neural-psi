# NeuralPSI — Validation & Evaluation (everything after C2)

*Companion to `C2_quantization_writeup.md`. C2 established the **bridge** (16 floats → 128-bit
Super-Bit code, 1.88% EER). This document covers what we did **after** that: proving the bridge works
through the **real cryptography** (not just our Python model), and measuring the system's **efficiency**
(communication, latency, storage) and the **dimension trade-off**.*

> **One-line summary.** The Super-Bit bridge now runs end-to-end through the **actual** flash-psi
> crypto and behaves exactly as our model predicted (real FAR **4.43e‑2** vs predicted **4.45e‑2**);
> the system's communication tracks Bui–Cong's published numbers (**~150 MB @ 100 K**, matching their
> 153 MB); storage is **16 B/user** (vs Blind-Touch's 1.67 KB + 117 MB key); and **d = 128 is the knee**
> of the accuracy/size trade-off.

---

## 1. Real-crypto validation — the headline result

**Why it matters.** Everything in our accuracy/operating-point work (Tier-1, Tier-2) was computed
through a *faithful Python port* of flash-psi plus a closed-form sub-sampling model `q(H)`. That left
one open question: **do our actual 128-bit codes behave the same way through the real masked-OPRF +
garbled-circuit + VOLE + Shamir protocol?** We closed it.

**What we did.** We integrated the `fingerprint.rs` harness into the working flash-psi crate (one small
API addition), then pushed our real held-out 224-model Super-Bit codes through the actual binary:
**a database of 100 enrollment codes, 100 probe queries → 100 genuine + 9,900 impostor decisions**, all
through real crypto, at our operating point (weight = 14, t = 2, T = 64).

**Result — the real crypto matches our model almost exactly:**

| metric | **REAL (crypto)** | **PREDICTED (our model)** |
|---|---|---|
| TAR | 100.00% | 99.52% |
| **FAR** | **4.43e‑2** | **4.45e‑2** |

And the accept-rate tracks the model across the **entire fractional transition**, not just the easy ends:

| Hamming bucket | real accept | predicted |
|---|---|---|
| [0,10) | 100.0% | 100.0% |
| [20,25) | 86.5% | 88.9% |
| [25,30) | 58.5% | 55.5% |
| [30,35) | 21.0% | 21.3% |
| [35,40) | 6.5% | 6.1% |
| [50,128] | 0.0% | 0.0% |

Confident regime: **94/94 accepts** and **8,522/8,523 rejects** correct.

**What this buys us:**
1. **The bridge is real-crypto-correct** — first end-to-end run of our own codes through the genuine
   FLPSI stack.
2. **Our entire evaluation is retroactively validated.** Every accuracy/FAR number we've reported came
   from the surrogate + `q(H)`; this proves that model is faithful (real FAR 4.43e‑2 vs predicted
   4.45e‑2, two significant figures). We can now defend all of it.

*(Detail: `realcrypto_results.md`. Script: `implementation/week1/09_realcrypto_validate.py`.)*

---

## 2. Efficiency evaluation (§7.3 communication, §7.4 latency, §7.5 storage)

Measured on the real protocol binary with network-IO tracking, sweeping the database size.

### §7.3 Communication — and it matches Bui–Cong

| N (database size) | 100 | 1,000 | 5,000 | 10,000 | 50,000 | → 100 K | → 1 M |
|---|---|---|---|---|---|---|---|
| communication | 0.40 MB | 1.76 MB | 7.77 MB | 15.28 MB | 75.40 MB | **~150 MB** | **~1.5 GB** |

Linear at **~1.54 KB/record**. Extrapolates to ~150 MB @ 100 K and ~1.5 GB @ 1 M — matching
Bui–Cong's reported **153 MB / 1533 MB**. The quantiser adds **nothing** to communication; it only sets
the 128-bit width the protocol already uses. Replacing HE removes Blind-Touch's 117 MB Galois key.

### §7.4 Latency breakdown — the bridge is not the bottleneck

| phase | N = 1,000 | N = 5,000 | scaling |
|---|---|---|---|
| CNN forward (client, GPU) | 0.80 ms | 0.80 ms | flat in N |
| Quantise (client) | 0.002 ms | 0.002 ms | flat in N |
| **FLPSI online (query-time)** | **187 ms** | **823 ms** | ~0.16 ms/record |
| FLPSI offline (preprocessing) | 553 ms | 1,946 ms | data-independent, off the query path |

Client-side work (CNN + the Super-Bit quantisation) is **under 1 ms** — the cost is the server-side
FLPSI scan. The offline setup is data-independent and can run before the query arrives.

### §7.5 Storage

| system | per-user template | one-time key | 512 users |
|---|---|---|---|
| Blind-Touch (HE/CKKS) | ~1.67 KB | 117 MB Galois key | ~855 KB + 117 MB |
| **NeuralPSI** | **16 B** | ~0 | **8 KB** |

**~107× smaller per template, and the 117 MB key is gone.**

*(Detail: `efficiency_results.md`.)*

---

## 3. Dimension ablation (§7.6) — why d = 128

We swept the code length and re-encoded the same held-out fingers:

| d (bits) | Super-Bit blocks | storage | EER | separation d′ |
|---|---|---|---|---|
| 64 | 4 | 8 B | 2.38% | 3.656 |
| **128** | **8** | **16 B** | **1.88%** | 3.778 |
| 256 | 16 | 32 B | 1.89% | 3.814 |

**Accuracy saturates exactly at d = 128.** 64→128 buys a real half-point; 128→256 buys nothing while
doubling the template. This is the 16-D information ceiling made concrete (128 sign projections already
capture the ~16 bits of genuine information). Bonus finding: **the FLPSI backend is hardwired to
d = 128** (a compiled `EQ128` garbled circuit + a 128-element OPRF key), so 128 is also the natively
supported width. **d = 128 is the knee** on all three counts: accuracy, storage, and backend support.

*(Detail: `dim_ablation_results.md`. Script: `implementation/week1/10_dim_ablation.py`.)*

---

## 4. Artifacts & how to reproduce

| Artifact | What |
|---|---|
| `implementation/flash-psi/src/bin/fingerprint.rs` | real FLPSI on real codes (built into the crate) |
| `implementation/week1/09_realcrypto_validate.py` | real-crypto validation (§1 above) |
| `implementation/week1/10_dim_ablation.py` | dimension ablation (§3 above) |
| `reports/realcrypto_results.md`, `efficiency_results.md`, `dim_ablation_results.md` | result tables |
| `reports/STATUS.md` | full project status + resume guide |

```bash
# real-crypto validation (GPU sandbox)
cd implementation/week1
LD_LIBRARY_PATH=/run/host/usr/lib64 ~/bt-gpu-sbx/bin/python 09_realcrypto_validate.py --batch 100 --queries 100
# dimension ablation
LD_LIBRARY_PATH=/run/host/usr/lib64 ~/bt-gpu-sbx/bin/python 10_dim_ablation.py
# communication / latency sweep
cd ../flash-psi
for M in 100 1000 5000 10000 50000; do ./target/release/simulation --sim-type flpsi -m $M -w 14 -s 64 -t 2 --track-io --csv; done
```

---

## TL;DR for the slide
- **Validated the bridge on real crypto:** our codes through the actual OPRF/GC/VOLE/Shamir binary —
  real FAR **4.43e‑2** vs predicted **4.45e‑2**. The model we trusted is faithful.
- **Efficiency measured:** ~**1.54 KB/record** comm (→ ~150 MB @ 100 K, matching Bui–Cong); **16 B/user**
  storage (vs 1.67 KB + 117 MB key); client-side CNN+quantise **< 1 ms**.
- **d = 128 justified:** accuracy saturates there; smaller loses accuracy, larger only wastes storage;
  and the backend is built for 128.
