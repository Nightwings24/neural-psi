# Neural-PSI

**Privacy-preserving fingerprint authentication that replaces homomorphic encryption with
Fuzzy-Labelled Private Set Intersection.**

A three-month internship project at QNu Labs by Rineet Pandey, Ayan Shil and Prabhudutta Prusti,
supervised by Dr. Amit Kumar Chauhan, Ayan Chattopadhyay and Dr. Rohitkumar R. Upadhyay.

```
fingerprint image ──CNN──► e ∈ ℝ¹⁶ ──Super-Bit──► b ∈ {0,1}¹²⁸ ──FLPSI──► label or reject
                  (client, plaintext)              (16 bytes)      (server never sees the code)
```

## The idea

Blind-Touch (Choi, Woo & Kim, AAAI 2024) matches fingerprints under CKKS homomorphic encryption.
Its accuracy comes from the CNN feature extractor, **not** from the encryption — so the matching
layer can be replaced. Fuzzy-Labelled PSI already solves approximate matching efficiently, but it
speaks **binary strings under Hamming distance** while a CNN emits **floating-point vectors**.

The bridge between them is this project's main technical contribution: a public, training-free
**Super-Bit quantiser** that turns 16 floats into a balanced 128-bit code whose Hamming distance
tracks angular similarity.

## Results

| Representation | Bits | Equal error rate |
|---|---|---|
| Floating-point embedding *(the ceiling HE matches on)* | — | **0.33%** |
| Naive `sign(>0)` | 16 | 10.12% |
| ITQ | 16 | 5.55% |
| **Super-Bit (chosen)** | **128** | **1.88%** (95% CI [1.56, 2.20]) |
| ITQ *(bit-budget-matched)* | 128 | 4.55% |

| | Blind-Touch (HE/CKKS) | Neural-PSI |
|---|---|---|
| Template per user | 1.67 KB | **16 B** |
| Evaluation (Galois) key | 117 MB | **none** |
| Latency, single server (N = 5,000) | 1,334 ms | **~824 ms** |
| Per-query communication (N = 5,000) | 856 KB | 7.77 MB |
| Output | match score | **record label** |

The protocol behaviour was validated against the **real** cryptographic binary, not only in
simulation: over 100 genuine and 9,900 impostor decisions, the measured false-accept rate was
**4.43e-2** against a predicted **4.45e-2**.

### The main research finding

That 4.4e-2 false-accept rate is not a bug — it is the system working as specified, and it is
**1,550× worse than the parameters were chosen for**. The FLPSI error analysis assumes impostor
codes are uniform (Hamming distance ~ Binomial(128, ½), σ = 5.66). Codes from a 16-dimensional
embedding are not: measured over 2.88M held-out pairs their mean is *exactly* 64.0 — so the
quantiser's bit-balancing makes the problem invisible to any first-moment check — while σ is
**20.22**. Worse, 8 of those pairs are *identical* 128-bit codes, an irreducible **2.78e-6**
floor that no choice of protocol parameters can beat. Retuning does not fix it; multi-finger
fusion buys four orders of magnitude.

Full analysis: [docs/results/operating-point.md](docs/results/operating-point.md) and
[docs/results/code-analysis.md](docs/results/code-analysis.md).

> **Scope.** All accuracy figures are on SOCOFing — an intentionally easy, single-dataset
> protocol whose probes are synthetic alterations of the same capture — under a **semi-honest**
> adversary model. Generalisation across capture conditions is untested. See
> [docs/results/](docs/results/).

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src/) | The pipeline. Library modules (`model.py`, `quantizer.py`, `flpsi_match.py`) plus numbered experiment scripts `01`–`13`. |
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
