# Submission manifest - Neural-PSI

Privacy-preserving fingerprint authentication by composing a learned feature extractor with
fuzzy-labelled private set intersection.

## Required deliverables

| Deliverable | File | Source | Pages |
|---|---|---|---|
| **Final technical report** | [`technical-report.pdf`](technical-report.pdf) | `paper/technical-report.tex` | 48 |
| **Final presentation slides** | [`slides.pdf`](slides.pdf) | `paper/slides.tex` | 29 frames |
| **Source code** | - | [`src/`](../src), [`crypto/`](../crypto), [`demo/`](../demo), [`baseline-blind-touch/`](../baseline-blind-touch) | - |
| **Simulation files / results** | - | [`docs/results/`](../docs/results), [`paper/figures/`](../paper/figures) | - |
| **Other project materials** | [`paper.pdf`](paper.pdf), [`pipeline-walkthrough.pdf`](pipeline-walkthrough.pdf) | `paper/paper.tex`, `paper/pipeline-walkthrough.tex` | 26, 19 |

The PDFs here are copies for convenience; the LaTeX sources live in [`paper/`](../paper) and are
the versions to edit.

### What each document is for

- **`technical-report.pdf`** - the full report: background on every primitive, the system design,
  the parameter analysis, the security proof, and the evaluation.
- **`paper.pdf`** - a self-contained conference-format (LNCS) paper covering the same work in
  26 pages, with the background compressed to cited preliminaries.
- **`slides.pdf`** - the presentation deck.
- **`pipeline-walkthrough.pdf`** - a tutorial walkthrough of the three computational stages with
  worked numerical examples, aimed at a reader new to the system.

The technical report and the paper `\input` the *same* files for the parameter analysis
(`paper/sections/05-parameters.tex`) and the security analysis
(`paper/sections/06-security.tex`), so the two documents cannot report different numbers.

## Headline results

| | |
|---|---|
| Equal-error rate, 128-bit Super-Bit code | **1.88%** (95% CI [1.56, 2.20]) |
| Floating-point ceiling | 0.33% |
| Server-side template | **16 B** (vs 1.67 KB for the HE baseline, 107× smaller) |
| Per-client key material | **none** (vs a 117 MB Galois key) |
| Measured per-record false-accept rate at the published operating point | **4.58e-2** - 1,550× the value the uniform-code model predicts |
| Irreducible code-collision floor | **2.78e-6** (95% CI [1.20e-6, 5.48e-6]) |

The last two rows are the project's main research finding: the FLPSI error analysis assumes
impostor codes are uniform, and codes derived from a 16-dimensional embedding are not - they are
perfectly balanced (so the assumed *mean* is exactly right and no first-moment check fires) while
carrying 3.6× the assumed standard deviation.

## Rebuilding

All four documents build with [Tectonic](https://tectonic-typesetting.github.io/), which fetches
its own TeX packages - no TeX Live installation needed:

```bash
cd paper
tectonic -X compile paper.tex
tectonic -X compile technical-report.tex
tectonic -X compile slides.tex
tectonic -X compile pipeline-walkthrough.tex
```

Regenerating the results and figures (see [`../README.md`](../README.md) for environment setup):

```bash
cd src
python 11_operating_point.py     # -> docs/results/operating-point.*  + 3 figures
python 12_code_analysis.py       # -> docs/results/code-analysis.json + 1 figure
python 13_cnn_figures.py         # -> the 7 CNN walkthrough figures
```

Every `.md` in `docs/results/` names the script that produces it.

## Datasets

**SOCOFing** (Sokoto Coventry Fingerprint Dataset) - 6,000 fingerprints from 600 subjects.
Publicly available for research use. Not committed to this repository; see the README for the
download step. All accuracy figures in every document are measured on SOCOFing, using an
identity-disjoint split of 4,800 training images and 1,200 held-out identities.

Cite: Y. I. Shehu et al., *Sokoto Coventry Fingerprint Dataset*, 2018.

## Third-party components

- The **Blind-Touch** baseline (Choi, Woo & Kim, AAAI 2024) is included under
  `baseline-blind-touch/` for comparison; its documentation describes that upstream project's own
  files.
- The **FLPSI** construction is due to Bui and Cong (2025). It is used as a black box; the
  construction, its correctness argument and its security proof are theirs, and the documents say
  so explicitly. Only `crypto/fingerprint.rs` and `crypto/neuralpsi-flash-psi.patch` are ours -
  the upstream implementation is not vendored, and `crypto/README.md` gives build instructions.
