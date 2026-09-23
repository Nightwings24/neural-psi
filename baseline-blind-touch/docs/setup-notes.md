# Blind-Touch demo - setup notes & deviations from the guide

This 3-container demo (`client` + `main` + `cluster1`) runs the **real repo code**
(Flask + HTTP + SEAL), not the guide's notebook-only server model. Below is every
place reality differed from `blind-touch-implementation-guide.md`, and what was done.

## Architecture
- The guide assumes the server is a Jupyter notebook watching a shared volume.
  **Reality:** the server is Flask apps - `server/main/` + `server/cluster1..3/` -
  that talk over HTTP on port 8090. There is no server notebook in the repo.
- We run **main + 1 cluster** (chosen for RAM headroom). `main/service.py` was
  trimmed from 3 clusters to 1: `urls = ['http://cluster1:8090/blindtouch']` and
  `result = ctxt1` instead of `add_many([ctxt1,ctxt2,ctxt3])`.

## Config paths (were placeholder strings in the repo - all filled in)
Shared volume `shared_ckks` is mounted at `/workspace/shared_data` in every container:
- `keys/`        public.key, secret.key (client only), galois.key, relin.key
- `models/`      feature_model, model  (TF SavedModel dirs, from training)
- `data/`        sokoto_real_224.npy (preprocessed), test_data.npy (testset)
- `ciphertexts/` ctxt1 (enrolled), target_enc (the query, saved by main)
- `results/`     clustering_1 (cluster output), compressed (main output)

Edited: `server/main/app.py`, `server/main/service.py`, `server/cluster1/conf.py`.

## Dockerfile.base fixes
- **Keras conflict:** requirements.txt pins both `tensorflow==2.11.0` and
  `keras==2.13.1`, which is contradictory (TF 2.11 needs its bundled keras 2.11).
  We install TF 2.11 only and use its bundled keras. Installing keras 2.13 breaks
  `import tensorflow`.
- **Missing deps:** `flask` and `aiohttp` are required by the server but are NOT
  in requirements.txt. Added.
- Ubuntu 20.04 already ships Python 3.8 as `python3`, so the guide's
  `update-alternatives` juggling is unnecessary and was dropped.
- SEAL built with `-j2` to avoid OOM on a memory-tight host; build fails fast with
  an `import seal` smoke test.

## Notebook fixes (client)
- **Added a key-generation cell.** The client notebook only *loaded* keys; it had
  no keygen (the guide's "run the key generation cells" referenced cells that don't
  exist). We now generate + save PK/SK/GK/RL into `shared_data/keys`.
- Added `from keras import backend as K` (metric fns used it without importing).
- Wired testset / model / ctxt paths and the main-server URL
  (`http://main:8090/blindtouch`).

## Notebook fixes (training)
- **Epochs reduced 150 -> 15.** 150 epochs on CPU is impractical (hours-days; the
  guide's "10-30 min" is wrong). Raise `EPOCHS` if you have a GPU. Accuracy will be
  low at 15 - fine for demonstrating the HE pipeline, not for real matching.
- Wired data path, model save paths, testset save path.

## Added: dataset preprocessing
- `training/preprocess_sokoto.py` converts SOCOFing `Real/*.BMP` -> the
  `sokoto_real_224.npy` array the training notebook loads. Neither the repo nor the
  guide included this step. Run it in the client container before training.

## Still required from you
- **Docker** must be installed (needs sudo - see the install block).
- **kaggle.json** at `~/.kaggle/kaggle.json` to download SOCOFing, OR drop the
  dataset under `~/blindtouch-demo/dataset/SOCOFing/Real/` manually.

## Run order (Phase 5)
1. `docker compose up -d client`  (only the client first)
2. In client Jupyter (http://localhost:8888, token `blindtouch`):
   a. `docker exec -it blindtouch-client python3 /workspace/training/preprocess_sokoto.py`
   b. run training notebook  -> saves models + testset
   c. run client notebook through the keygen + enrollment (ctxt) cells -> keys + ctxt1
3. `docker compose up -d main cluster1`  (now that keys/models/ctxt exist)
4. run the final auth cell in the client notebook -> Authenticated / Rejected
