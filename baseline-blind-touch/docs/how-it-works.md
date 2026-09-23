# Blind-Touch - Clarifications

A Q&A companion to the main docs, capturing the conceptual questions worked through
about how this demo actually does homomorphic-encryption (HE) fingerprint matching.
For *setup/run* steps see [`reproduce.md`](reproduce.md); for architecture see [`architecture.md`](architecture.md).

---

## 1. How does the server compare the ciphertext? (the "do it blindly" part)

The comparison lives in `server/cluster1/service.py` → `start_clustering()`. The key
idea: the server **never compares two single fingerprints**. Thanks to CKKS's SIMD
packing, it scores the query against **all 512 enrolled templates at once**, and the
whole answer comes back as a single ciphertext.

### Data layout (why the math looks the way it does)

CKKS encrypts a *vector* of **8192 slots** (`slot_count = poly_modulus_degree/2 =
16384/2`). That vector is carved into **512 blocks of 16 slots**:

- **16 = the feature-embedding dimension** (the neural net boils a fingerprint down to
  16 numbers). That's why the one-hot mask and the rotation-sum both span 16.
- **512 = number of enrolled identities**, packed side by side.

So one ciphertext holds 512 templates, and one homomorphic evaluation produces 512
scores in parallel. That batching is the whole performance trick.

- `clustering_ctxt` (loaded from `ctxt1`) = the enrolled gallery - 512 templates.
- `target_enc` = the client's query feature vector, replicated across all 512 blocks so
  every template is compared to the same probe.

### The comparison, step by step (`start_clustering`)

```python
sub_ctxt = evaluator.sub(ctxt, clustering_ctxt)                 # (1) query − template, per slot
result   = fc1_layer(evaluator, square(evaluator, sub_ctxt))    # (2) + (3)
```

1. **Difference** - `sub` gives `q − t` for all 8192 slots (all 512 templates) in one op.
2. **Square** (`square()`) - `multiply(ctxt, ctxt)` + relinearize → element-wise
   `(q − t)²`. Double duty: it's the **squared-distance** term *and* the HE-friendly
   polynomial activation. (CKKS can't do ReLU/swish, so the network was trained with a
   custom `square` activation precisely so it can be evaluated under encryption.)
3. **Learned scoring layer** (`fc1_layer()`) - turns the 16 squared-differences in each
   block into one match score:
   - Multiply by the **trained final-dense-layer weights** (`model.get_weights()[32]`,
     replicated ×512 so every block uses the same learned weights). Now each slot holds
     `wᵢ·(qᵢ−tᵢ)²`.
   - **Rotation-sum**: rotate by 1, 2, 4, 8 and add each time - a log-reduction that
     accumulates all 16 slots of a block into the block's first slot →
     `Σᵢ wᵢ(qᵢ−tᵢ)²`, one weighted squared-distance per identity.
   - **Add bias** (`model.get_weights()[33]`).
   - **Multiply by the one-hot mask** `[1,0,…,0]×512` → zeroes the 15 junk slots in each
     block, leaving the clean score in slot 0 of each of the 512 blocks.

So, entirely on ciphertext, the server computes a **learned distance metric**:

> **scoreᵢ = bias + Σⱼ wⱼ · (queryⱼ − templateᵢⱼ)²**  for each enrolled identity *i*

The trained weights/bias were fit so a genuine match scores near 1 and impostors stay low.

### Plumbing you can ignore conceptually

`rescale_to_next_inplace`, `mod_switch_to_inplace`, `relinearize_inplace` are **not**
part of the comparison - they're CKKS bookkeeping (keeping ciphertext scales aligned and
managing the noise/modulus budget after each multiply). The final
`rotate_vector(result, -(CLUSTER_NUM-1))` is a **no-op** in this demo (`CLUSTER_NUM=1`);
it only matters when stitching multiple clusters' partial results.

### What goes back to the client

`cluster1` saves the encrypted result; `main`'s `start_clustering` just forwards it (with
3 clusters it would `add_many` the partials). **The server never decrypts** - it holds
only the public / galois / relin keys, no secret key.

---

## 2. ELI5 - what happens when a person X gives their fingerprint?

Analogy: X's fingerprint becomes a **secret padlocked box** only X can open.

1. **X presses the scanner (client).** The trained neural net turns the fingerprint image
   into a short list of **16 numbers** - the fingerprint's "essence." The image itself is
   discarded.
2. **The client locks those 16 numbers in a box (encryption).** CKKS scrambles them into a
   ciphertext. Only X's **secret key** can open it - and the client never gives that key away.
3. **The locked box is mailed to the server** (`target_enc`). The server holds it but
   **cannot see inside** - no secret key. It's gibberish to the server.
4. **The server compares boxes without opening them (the magic).** The server has a shelf
   of locked boxes - one per enrolled person (512). HE lets it do math on locked boxes so
   the answer comes out as a *new locked box*. For everyone it computes
   "how different is X's box from this person's?" (subtract → square → learned score). Out
   pops a locked box of **similarity scores**, computed **blindfolded**.
5. **The locked score-box is mailed back** to the client. Still locked.
6. **Only X can open it (decryption + decision).** The client uses X's secret key, reads
   the scores, takes the best one, and checks: **is it above 0.99?** Yes → "Authenticated,
   this is X." No → "Rejected."

**One sentence:** X's fingerprint becomes 16 secret numbers locked in a box; the server
does all the matching math *on the locked box without opening it*, produces a locked
"score" box, and only X (at the client, with the secret key) unlocks it to learn yes/no.

**Why it matters:** if a hacker steals everything on the server, they get a shelf of
locked boxes and no key. The fingerprints never exist in readable form outside the client.

---

## 3. Does the server send back all 512 score boxes?

**No - it sends back exactly ONE box,** which has 512 compartments inside it.

CKKS doesn't lock one number in a box; it locks a **whole vector of 8192 slots** in a
single ciphertext. The server arranged all 512 comparisons into different slots of that
one vector. So when the math finishes, the 512 scores all live in **different slots of one
ciphertext** (slots 0, 16, 32, 48, …). That one box is mailed back.

At the client:
- The secret key opens the **single box once** (one decryption).
- Out comes the full 8192-number vector.
- Read slots 0, 16, 32, … → the 512 similarity scores.
- Scan for the highest; check against 0.99.

| You might picture | What actually happens |
|---|---|
| 512 separate boxes mailed back | **1 box** mailed back |
| Open 512 boxes | **1 decryption** |
| 512 scores | 512 scores **packed in the slots of that one box** |

This is *why* it's fast: one subtract, one square, one weighted-sum - each hitting all
8192 slots at once (SIMD) - compares against all 512 people simultaneously, and the whole
answer comes home in a single ciphertext.

*Footnote:* with 3 clusters the gallery is split across them and `main` `add_many`s the
three partial boxes into one before sending. In this 3-container demo there's only
`cluster1`, so its one box *is* the final box.

---

## 4. Why is the `shared_data/` folder completely empty?

Because **the containers don't use that folder.**

- The host folder `./shared_data/` is leftover scaffolding (empty subdirs) - vestigial.
- The real data lives in a Docker **named volume** `shared_ckks`, mounted at
  `/workspace/shared_data` inside every container (`docker-compose.yml`).

A named volume is **not** a host bind-mount - Docker stores it under
`/var/lib/docker/volumes/blindtouch-demo_shared_ckks/_data`, not in the project folder. So
container writes go to the volume and `./shared_data/` stays empty forever.

The volume is fully populated:

```
models/       feature_model + model        (the val_acc-0.9958 SavedModels)
keys/         galois (136MB), public, relin, secret
ciphertexts/  ctxt1 / ctxt2 / ctxt3 + target_enc
data/         sokoto_real_224.npy + test_data.npy
results/      clustering_1 + compressed
```

| | Host `./shared_data` | Docker volume `shared_ckks` |
|---|---|---|
| Mounted into containers? | **No** | Yes, at `/workspace/shared_data` |
| Contents | empty dirs (vestigial) | all real keys/models/ciphertexts |
| Where on disk | the project folder | `/var/lib/docker/volumes/…` |

To make the files appear in the project folder, switch the compose mounts from the named
volume to a bind-mount (`./shared_data:/workspace/shared_data`). Optional - the demo works
as-is.

---

## 5. File map of `blindtouch-demo/`

**Mental model:** `docker/` + compose *build the world* → `training/` + `dataset/` *make
the model* → `client/` + `server/` *are the live app* → the `shared_ckks` volume *is the
shared disk* → `docs/` + `report/` *explain and record what happened*.

### Docker build & orchestration
| File | Role |
|---|---|
| `docker-compose.yml` | Defines the 3 services (`client`, `main`, `cluster1`) + the `shared_ckks` volume. |
| `docker/Dockerfile.base` | Heavy base image: Microsoft SEAL, TensorFlow/Keras, Flask, OpenCV (incl. the OpenCV-headless fix). |
| `docker/Dockerfile.{client,main,cluster1}` | Thin per-service images built on the base. |
| `requirements.txt` | Pinned Python deps. |

### Application code
| Path | Role |
|---|---|
| `client/Blind-Touch-Client.ipynb` | Original full client notebook (keygen → enroll → auth). |
| `client/Blind-Touch-Enroll.ipynb` | Split: keygen + encrypt templates, no network. Run first. |
| `client/Blind-Touch-Auth.ipynb` | Split: load keys, encrypt a query (`BT_AUTH_IDX`), POST, decrypt, print Authenticated/Rejected. |
| `server/main/{app.py,service.py}` | Front door: receives encrypted query, fans out, returns encrypted result. |
| `server/cluster1/{app.py,service.py,conf.py}` | Worker doing the blind comparison math. |

### Training & data
| Path | Role |
|---|---|
| `training/Blind-Touch-Training-SampleNotebook(SOKOTO).ipynb` | The notebook actually trained with (150 epochs, val_acc 0.9958). |
| `training/Blind-Touch-Training-SampleNotebook(PolyU).ipynb` | Alternate dataset version - unused. |
| `training/preprocess_sokoto.py` | Raw BMPs → `sokoto_real_224.npy`. |
| `dataset/SOCOFing/` | ~844 MB, 55,270 BMP fingerprints (Real + Altered) - raw SOCOFing dataset (git-ignored; download from Kaggle). |

### Data store
| Path | Role |
|---|---|
| `shared_data/` | Empty/vestigial host folder. Real data is in the `shared_ckks` Docker volume. |

### Docs & report
| File | Role |
|---|---|
| `README.md` | Project overview and quickstart (the front door). |
| `docs/how-it-works.md` | This doc - conceptual Q&A. |
| `docs/architecture.md` | Plain-language architecture write-up. |
| `docs/reproduce.md` | Step-by-step reproduce / recovery commands. |
| `docs/setup-notes.md` | Every deviation from the upstream guide and why. |
| `docs/implementation-report.md` | Implementation report (markdown source). |
| `docs/implementation-report.pdf` | Rendered PDF of the report. |
| `report/build_report.py` | Builds the PDF from `docs/implementation-report.md` + the evidence assets. |
| `report/assets/ev_*.txt` | Captured console evidence (training / servers / auth) embedded in the report. |

> Run logs from the reproduction (`*.log`) and the trained models / keys / ciphertexts are
> **not** committed - they live in the Docker volume or are git-ignored. See [`.gitignore`](../.gitignore).
