# Neural-PSI

**Privacy-preserving fingerprint authentication that replaces homomorphic encryption with
Fuzzy-Labelled Private Set Intersection.**

A three-month internship project at QNu Labs by Rineet Pandey, Ayan Shil and Prabhudutta Prusti,
supervised by Dr. Amit Kumar Chauhan, Ayan Chattopadhyay and Dr. Rohitkumar R. Upadhyay.

```
fingerprint image ──CNN──► features ──learned binariser──► b ∈ {0,1}¹²⁸ ──FLPSI──► label or reject
                  (client, plaintext)                       (16 bytes)      (server never sees the code)
```

> ### 🔎 Latest results — two improvement iterations
> Since the initial Super-Bit design, two rounds of work on the binariser cut same-sensor EER
> from **1.89% → 0.17%** (~11×, on SOCOFing) and communication by **45%**, on the same 128-bit
> backend. **Start here:**
> - **[docs/results/RESULTS-SUMMARY.md](docs/results/RESULTS-SUMMARY.md)** — one-page overview (both iterations, headline table, Blind-Touch comparison).
> - **[docs/results/improvement-findings.md](docs/results/improvement-findings.md)** — full technical record.
> - **[docs/paper/architecture.html](docs/paper/architecture.html)** — end-to-end architecture diagram (open in a browser).
> - **[docs/results/future-directions.md](docs/results/future-directions.md)** — what to try next.

## The idea

Blind-Touch (Choi, Woo & Kim, AAAI 2024) matches fingerprints under CKKS homomorphic encryption.
Its accuracy comes from the CNN feature extractor, **not** from the encryption — so the matching
layer can be replaced. Fuzzy-Labelled PSI already solves approximate matching efficiently, but it
speaks **binary strings under Hamming distance** while a CNN emits **floating-point vectors**.

The bridge between them is this project's main technical contribution: a public, training-free
**Super-Bit quantiser** that turns 16 floats into a balanced 128-bit code whose Hamming distance
tracks angular similarity.

## Results

**Same-sensor EER on SOCOFing (held-out, identity-bootstrap 95% CI) across two improvement
iterations — same 128-bit FLPSI backend throughout:**

| Binariser | EER (Altered-Easy) | impostor σ (ideal 5.66) |
|---|---|---|
| Super-Bit LSH *(initial)* | 1.89% [1.57, 2.22] | 20.1 |
| Round 1 — metric-aware ortho-thermometer | 0.87% [0.62, 0.99] | 14.6 |
| **Round 2 — quantization-aware feature-head** | **0.17% [0.06, 0.26]** | **6.72** |

The Round-2 code also cuts **communication by 45%** at a matched operating point (validated
end-to-end through the real crypto binary) and holds its advantage across the Altered
Easy/Medium/Hard ladder. Full numbers + generality tests (PolyU, FVC2002):
**[docs/results/RESULTS-SUMMARY.md](docs/results/RESULTS-SUMMARY.md)**.

### vs Blind-Touch (the HE baseline we build on)

| | Blind-Touch (HE/CKKS, 3-server) | Neural-PSI (FLPSI, 2-party) |
|---|---|---|
| SOCOFing EER | 0.7% | **0.17%** |
| Template per user | 1.67 KB | **16 B** |
| Evaluation (Galois) key | 117 MB | **none** |
| Per-query communication (N = 5,000) | **856 KB** | 2.38 MB *(was 7.77 MB)* |
| Output | match score | **record label only** |

We win on accuracy and protocol simplicity (no homomorphic-encryption cluster); compressed CKKS
still beats us on raw communication — the −45% narrows, not erases, that gap. All figures cross-
checked against the **real** cryptographic binary, not just simulation.

### The diagnosis that drove the improvements

The gains trace to one measurement: a 128-bit code built from a 16-dimensional embedding uses
only **~14–18 of its bits**. Impostor codes are far from the uniform ideal the FLPSI error
analysis assumes — mean *exactly* 64.0 (invisible to any first-moment check) but σ = 20.1 vs the
Binomial(128, ½) ideal of 5.66, and a few distinct fingers even produce identical codes.
**Round 1** fixed the *distance metric* the code preserves (Euclidean, not cosine); **Round 2**
rebuilt the code from the CNN's richer features to be balanced and decorrelated, pushing σ to
**6.72** and closing most of the accuracy/communication gap at once. Full analysis:
[docs/results/improvement-findings.md](docs/results/improvement-findings.md).

> **Scope.** All accuracy figures are on SOCOFing — an intentionally easy, single-dataset
> protocol whose probes are synthetic alterations of the same capture — under a **semi-honest**
> adversary model. Generality tests on PolyU and FVC2002 show the communication/code-quality
> gain is universal, while the large accuracy gain needs a high-fidelity corpus; cross-corpus
> transfer is a known-hard wall. See [docs/results/](docs/results/).

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src/) | The pipeline. Library modules (`model.py`, `quantizer.py`, `flpsi_match.py`), the original experiment scripts `01`–`13`, and the two improvement iterations in `14`–`32` (dim sweep, metric-aware bridge, fusion, fragile-bit diagnostic, quantization-aware feature-head, operating-point/comms, generality on PolyU/FVC/DeepPrint). |
| [`crypto/`](crypto/) | Our additions to the flash-psi protocol: a patch and the `fingerprint` binary that runs real FLPSI on real codes. Upstream is **not** vendored. |
| [`demo/`](demo/) | Interactive Streamlit walkthrough of the whole pipeline, running the real model and (optionally) the real cryptography. |
| [`docs/`](docs/) | Explanations and measured results. |
| [`paper/`](paper/) | LaTeX sources: the LNCS paper, the technical report, the slide deck, the pipeline walkthrough and the presentation script. |
| [`submission/`](submission/) | **Built PDFs of every deliverable, with [`MANIFEST.md`](submission/MANIFEST.md).** |
| [`baseline-blind-touch/`](baseline-blind-touch/) | Our reproduction of the Blind-Touch HE baseline (Docker, notebooks, docs). |

## Getting started

```bash
git clone https://github.com/Nightwings24/neural-psi.git
cd neural-psi

python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
pip install -r requirements.txt
```

**The dataset and trained weights are not in the repository** (size and licensing). To reproduce:

1. Download **SOCOFing** (Shehu et al., 2018) and place it at `dataset/SOCOFing/`.
2. Extract the arrays and train:
   ```bash
   python src/01b_extract_dir.py --img 224     # -> src/data/x_real_224.npy, ...
   python src/train_gpu_224.py                 # -> src/data/feature_model_224.pt
   ```
3. Reproduce the accuracy ladder:
   ```bash
   python src/07_superbit_eer.py --img 224 --ckpt feature_model_224.pt
   ```
4. Reproduce the improvements. The diagnostic and Round-1 scripts run on the **committed** cached
   embeddings (`src/data/dim_ablation_emb_224.npz`), so they work without re-extracting SOCOFing;
   the Round-2 feature-head extracts CNN features from the raw images, so it needs step 1+2 done.
   ```bash
   python src/22_fragile_bits.py                         # diagnosis: effective bits, impostor σ
   python src/21_bridge_ci.py                            # Round-1 ortho-thermometer EER + CI
   python src/28_qat_featurehead.py --tag repro          # Round-2 feature-head (needs raw images)
   python src/29_qat_ladder.py --map data/qat_head_repro_map.npz --tag repro   # Easy/Med/Hard
   python src/25_qat_downstream.py --codes data/qat_codes_repro.npz --tag repro # comms vs baseline
   ```
   Generality scripts `30`–`32` need the PolyU / FVC2002 corpora and (for `32`) the DeepPrint
   weights — see [`docs/results/future-directions.md`](docs/results/future-directions.md).

For the real-cryptography measurements, build the protocol binary first — see
[`crypto/README.md`](crypto/README.md).

### Run the demo

```bash
cd demo && streamlit run app.py     # http://localhost:8501
```

Requires `src/data/feature_model_224.pt`. Full instructions in [`demo/README.md`](demo/README.md).

## Documentation

**Start here:** [`docs/quantization-bridge.md`](docs/quantization-bridge.md) — the core contribution,
explained end to end.

| Document | What it covers |
|---|---|
| [cnn-explainer.md](docs/cnn-explainer.md) | The feature extractor from scratch, for a non-specialist audience |
| [quantization-bridge.md](docs/quantization-bridge.md) | Why binarisation is needed, methods evaluated, the Super-Bit pipeline |
| [validation-and-evaluation.md](docs/validation-and-evaluation.md) | Real-crypto validation, efficiency, and the dimension ablation |
| [results/accuracy-eer.md](docs/results/accuracy-eer.md) | The EER ladder and the retraining that produced it |
| [results/operating-point-and-fusion.md](docs/results/operating-point-and-fusion.md) | Tuning `(w, t, T)`, and multi-finger fusion |
| [results/real-crypto-validation.md](docs/results/real-crypto-validation.md) | Real protocol vs the closed-form model |
| [results/efficiency.md](docs/results/efficiency.md) | Communication, latency, storage |
| [results/dimension-ablation.md](docs/results/dimension-ablation.md) | Why 128 bits |

## Built on

- **Blind-Touch** — Choi, Woo & Kim, *AAAI 2024*. The CNN architecture and the HE baseline.
- **Fuzzy-Labelled PSI** — Uzun et al., *USENIX Security 2021* (definition); Bui & Cong,
  *CANS 2025* (the Vector Ring-OLE construction and the implementation we use).
- **Super-Bit LSH** — Ji, Li, Yan, Zhang & Tian, *NIPS 2012*.
- **SOCOFing** — Shehu et al., 2018.

## Licence

MIT — see [LICENSE](LICENSE). Third-party components keep their own terms.
