# Pipeline source

Library modules plus the numbered experiment scripts, in the order they were run.
Every script writes to `src/data/` (git-ignored) and expects to be run from this directory.

## Library modules

| File | What |
|---|---|
| `model.py` | The Siamese CNN. `FeatureModel` (5 conv blocks → FC-16) and `SiameseModel` (the training head). |
| `quantizer.py` | **The contribution.** `NeuralPSIQuantizer` — centre → Super-Bit projection → median threshold → 128-bit code. Pure NumPy, no training. |
| `flpsi_match.py` | Faithful Python port of the flash-psi sub-sampling decision, for fast evaluation without the Rust binary. |
| `flpsi_surrogate.py` | Statistical FPR/FNR model of the FLPSI matcher. |

## Experiment scripts

| Script | Purpose |
|---|---|
| `01_extract.py` / `01b_extract_dir.py` | Build image arrays from SOCOFing (zip / unzipped directory). `01b --img 224` is the one used. |
| `02_train.py` / `train_gpu_224.py` | Train the Siamese CNN. `train_gpu_224.py` is the production run (224 px, 150 epochs). |
| `train_arcface_224.py` | Angular-margin variant. **Kept for reference — it collapses**: 4,800 finger classes cannot be prototypes in a 16-D embedding. |
| `03_eval_eer.py` | Baseline EER on float and naive-sign representations. |
| `recover_head_w1.py` | Recovers the head weights `w1` needed for the (rejected) metric-whitening option. |
| `04_tune_psi.py` / `05_export_psi_input.py` / `06_superbit_export.py` | Operating-point sweep and code export for the protocol. |
| **`07_superbit_eer.py`** | **The accuracy ladder** — float / sign / ITQ / Super-Bit in one run. |
| `08_tier2_fusion.py` | Operating-point re-tune and multi-finger fusion analysis. |
| **`09_realcrypto_validate.py`** | **Real-crypto validation** — pushes actual codes through the real protocol binary and compares against the closed-form model. |
| `10_dim_ablation.py` | EER vs code length (64 / 128 / 256). |
| **`11_operating_point.py`** | **The operating-point frontier** — measures the real impostor Hamming distribution, computes FAR/FRR exactly over a 6,848-point (w, t, θ) sweep, finds the irreducible collision floor, and measures multi-finger fusion. Writes 3 figures. |
| **`12_code_analysis.py`** | **What the code contains** — bit balance, inter-bit dependence, empirical q̂(H) vs the hypergeometric model, ITQ at a matched bit budget, and EER with identity-level bootstrap CIs across all three alteration levels. |
| `13_cnn_figures.py` | Renders the stage-by-stage CNN figures used by the pipeline walkthrough. |
| `run_flpsi_match.py` / `benchmark_timing.py` / `show_pipeline.py` | Match verification, timing, and a step-by-step pipeline walkthrough. |

## Typical run order

```bash
python 01b_extract_dir.py --img 224                          # dataset -> arrays
python train_gpu_224.py                                      # -> data/feature_model_224.pt
python 07_superbit_eer.py --img 224 --ckpt feature_model_224.pt
python 10_dim_ablation.py
python 09_realcrypto_validate.py --batch 100 --queries 100   # needs the Rust binary
```

Scripts 09 and `benchmark_timing.py` need the `fingerprint` binary — see
[`../crypto/README.md`](../crypto/README.md), or set `FLASH_PSI_BIN`.
