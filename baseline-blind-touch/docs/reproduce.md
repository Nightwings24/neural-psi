# Reproducing the demo, end to end

Exact, copy-pasteable commands to run the full Blind-Touch pipeline on one machine —
**preprocess → train → enroll → start servers → authenticate** — and the recovery steps for
the handful of things the upstream code doesn't set up for you.

For *why* this deployment differs from the upstream materials, see [`setup-notes.md`](setup-notes.md).
For the conceptual walkthrough, see [`architecture.md`](architecture.md) and [`how-it-works.md`](how-it-works.md).

> **Verified result:** training reached **val_acc 0.9958**; a genuine query (identity #467) scored
> **0.9984** under encryption — top match, over the 0.99 threshold — in ~0.23 s of homomorphic
> matching. → **Authenticated.**

---

## Prerequisites

- **Docker + Docker Compose.**
- **~14 GB RAM** recommended. A full 150-epoch CPU train takes several hours; lower `BT_EPOCHS`
  for a fast functional demo (accuracy will be lower, but the encrypted pipeline still works).
- **The SOCOFing dataset.** Download [SOCOFing from Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)
  and unzip so the real images live at `./dataset/SOCOFing/Real/` (6 000 `.BMP` files). The
  `dataset/` folder is git-ignored and never committed.

---

## STEP 0 — Build the base image

The three service images are built `FROM blindtouch-base:latest`, so build the base first
(cached layers make re-runs fast):

```bash
docker build -t blindtouch-base:latest -f docker/Dockerfile.base .
# Smoke test — every dependency must import cleanly:
docker run --rm blindtouch-base:latest \
  python3 -c 'import seal, tensorflow as tf, keras, flask, aiohttp, cv2; print(tf.__version__, keras.__version__)'
```

## STEP 1 — Build the service images

```bash
docker compose build
```

## STEP 2 — Bring up only the client

`main` and `cluster1` load keys/models at import time, so they only start cleanly *after* the
volume is populated. Start the client alone first:

```bash
docker compose up -d client
docker compose ps
```

## STEP 3 — Preprocess the dataset

```bash
docker exec blindtouch-client python3 /workspace/training/preprocess_sokoto.py
# → writes /workspace/shared_data/data/sokoto_real_224.npy   (uint8, (6000,224,224,1))
```

## STEP 4 — Train the Siamese model (the long pole)

```bash
docker exec -e BT_EPOCHS=150 blindtouch-client jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  '/workspace/training/Blind-Touch-Training-SampleNotebook(SOKOTO).ipynb' \
  --output executed-training.ipynb
# Outputs: models → shared_data/models/{feature_model,model}, testset → shared_data/data/test_data.npy
```

Set `BT_EPOCHS` to control length (150 = the full run; try 15 for a quick demo). Watch RAM —
training is the heavy step. The SavedModel is written only at the **end** of training, so an
interrupted run leaves no model.

## STEP 5 — Create the runtime directories (required — see Gotchas)

The upstream code never `mkdir`s these on the volume; without them `.save()` throws a SEAL
`I/O error` that surfaces *misleadingly* on the client as `incompatible version`.

```bash
docker exec blindtouch-client  mkdir -p /workspace/shared_data/ciphertexts /workspace/shared_data/keys
```

## STEP 6 — Enroll: generate keys + encrypt templates

```bash
docker exec blindtouch-client jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  '/workspace/client/Blind-Touch-Enroll.ipynb' --output executed-enroll.ipynb
# → keys/ (galois, public, relin, secret) + ciphertexts/ctxt1 on the shared volume
```

## STEP 7 — Start the HE servers

```bash
docker exec blindtouch-cluster1 mkdir -p /workspace/shared_data/results   # required (see Gotchas)
docker compose up -d main cluster1
docker compose logs --tail=30 cluster1   # should load model + keys, print "Number of slots: 8192"
docker compose logs --tail=30 main
```

## STEP 8 — Authenticate

```bash
docker exec -e BT_AUTH_IDX=467 blindtouch-client jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  '/workspace/client/Blind-Touch-Auth.ipynb' --output executed-auth.ipynb
# → prints the top match scores and "RESULT: Authenticated" / "Rejected"
```

`BT_AUTH_IDX` (0–511) picks which enrolled identity to present; omit it for a random one.

---

## Gotchas fixed during reproduction

- **Uncreated runtime directories** — `ciphertexts/`, `keys/`, and `results/` are never made by
  the code. Their absence throws a SEAL `I/O error` that the client mis-reports as
  `incompatible version`. Create them before enroll/auth (STEP 5 & 7).
- **Dependency conflicts** — `requirements.txt` pinned mutually incompatible TF/Keras versions and
  omitted `flask`/`aiohttp`; a transitive OpenCV broke `import cv2`. All pinned/added in the base
  image — see [`setup-notes.md`](setup-notes.md).
- **No dataset preprocessing / no keygen existed** — both were added (`preprocess_sokoto.py`; a
  keygen cell in the enroll notebook).
- **Server is Flask/HTTP, not notebook-watching** — the topology here reflects the actual code.

## Verification

```bash
docker exec blindtouch-cluster1 ls -la /workspace/shared_data/ciphertexts/
docker exec blindtouch-client   du -sh /workspace/shared_data/ciphertexts/*
docker exec blindtouch-main      ping -c2 cluster1
```

## Quick reference

```bash
docker compose up -d                 # bring everything up
docker compose down                  # stop (keeps the volume)
docker compose down -v               # stop + WIPE the shared volume (keys/models/ciphertexts!)
docker exec -it blindtouch-client bash
```

## Architecture / topology notes

- This 3-container demo (`client` + `main` + `cluster1`) runs the **real repo code** (Flask + HTTP
  + SEAL), not the upstream guide's notebook-only server model.
- `main/service.py` was trimmed from the paper's multi-cluster setup to one:
  `urls = ['http://cluster1:8090/blindtouch']` and `result = ctxt1` instead of
  `add_many([ctxt1, ctxt2, ctxt3])`. Scaling out is configuration-only.
- The client notebook was split into **Enroll** (keygen → encrypt templates, no network) and
  **Auth** (load keys → encrypt a query → POST → decrypt → decide) so the steps can run headless
  and independently.
