# Neural-PSI — presentation demo

An interactive Streamlit walkthrough of the complete pipeline, built on the real trained
model, the real quantiser, and (optionally) the real cryptographic protocol binary.

```
fingerprint image -> CNN -> e in R^16 -> Super-Bit -> b in {0,1}^128 -> FLPSI -> match or reject
```

## Run it

```bash
cd demo
streamlit run app.py
```

Then open the URL it prints (default <http://localhost:8501>). Press `Ctrl+C` to stop.

The app runs on CPU and takes about seven seconds to warm up on first load (feature extractor,
quantiser calibration, enrolment); everything is cached afterwards, so start it a minute before
presenting.

### Environment

Python 3.10+ with a CPU build of PyTorch is enough:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## What it shows

**Server database tab.** The enrolled records. Each finger is stored as a single 128-bit
code (16 bytes) — no image, no embedding, no key.

**Authentication tab.** Pick a scenario and press *Run authentication*:

- *Genuine* — an enrolled finger, presented as a fresh capture (SOCOFing Altered variants
  standing in for a new, imperfect scan), claiming its own identity.
- *Impostor* — a finger that never enrolled, claiming an enrolled user's identity.

The run then unfolds stage by stage: the captured images, the 16-D embedding as a bar chart,
the 128-bit code as a bit grid, and the PSI decision with Hamming distances to every enrolled
subject plus a bit-level diff against the closest record.

**How it works tab.** One-paragraph explanations of each stage and the measured results.

## Suggested narrative for the meeting

1. Open **Server database** — "this is everything the server holds: 16 bytes per finger."
2. **Authentication → Genuine** → Run. Walk through Stage 1 (16 floats), Stage 2 (128 bits),
   Stage 3 (the match). Point out genuine Hamming distance around 8 of 128 versus roughly
   64 for everyone else.
3. Switch to **Impostor** → Run. Access denied.
4. Optional: raise **Mask weight (w)** in the sidebar and re-run. Stricter masks reject more,
   but also start rejecting genuine users — the accuracy/security trade-off, live.
5. The credibility point: set **Stage 3 backend** to *Real cryptography (flash-psi)*
   and re-run. The same decision is now produced by the actual masked-OPRF, garbled-circuit,
   VOLE and Shamir protocol, in roughly 100 ms.

## Operating point

The demo runs **1:1 verification**: the user presents a finger *and claims an identity*, and the
protocol verifies the finger against that claimed record. Defaults are `w = 14, t = 2, T = 64` —
the operating point validated in the report (true-accept ~99%, false-accept ~4.5% per comparison,
confirmed against the real protocol binary).

Note that 1:N identification — searching every enrolled record — is a harder problem: a
per-comparison false-accept rate compounds with database size, so it needs either a stricter
operating point or several fingers per user. That is noted as future work, not demonstrated here.

Every slider is live, so the trade-off can be shown during the talk.

## Files

| File | Purpose |
|---|---|
| `app.py` | the Streamlit interface |
| `pipeline.py` | backend: dataset indexing, CNN embedding, Super-Bit quantiser, PSI decision |
| `.streamlit/config.toml` | light theme |

## What it depends on

- `../dataset/SOCOFing/` (download separately — see the root README) — Real captures for enrolment, Altered-Easy for probes.
- `../src/` — `model.py`, `quantizer.py`, and `src/data/feature_model_224.pt`
  (the 224 x 224, 150-epoch model) plus `src/data/ids_test_224.npy` for the held-out split.
- the `fingerprint` binary — only for the real-cryptography option. See `../crypto/README.md`;
  set `FLASH_PSI_BIN` if you build it elsewhere.

The quantiser is fitted on 300 identities that are **not** in the held-out test set, so no
identity shown in the demo influenced the public constants.

## Honest caveats to state if asked

- Accuracy figures come from SOCOFing Altered-Easy, an intentionally easy, single-dataset
  protocol. Cross-sensor evaluation is future work.
- Timings are single-host; there is no wide-area network round trip in these numbers.
