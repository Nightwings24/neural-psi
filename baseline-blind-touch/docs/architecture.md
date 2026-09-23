# Blind-Touch - Plain-Language Architecture Explainer

A surface-level walkthrough of the Blind-Touch privacy-preserving fingerprint
authentication system, written for explaining the project to a mentor. Companion
to [`implementation-report.md`](implementation-report.md) (which has the formal results).

---

## 1. The one-sentence version

> **Blind-Touch lets a server check whether your fingerprint matches an enrolled
> user - without the server ever being able to *see* your fingerprint.** It does
> the matching math directly on *encrypted* data.

## 2. The problem it solves

Normally, to match a fingerprint, a server stores your actual fingerprint template
and compares it to the one you present. If that server is hacked, your fingerprint
leaks - and unlike a password, **you can't reset your fingerprint.** It's
compromised forever.

Blind-Touch fixes this with **homomorphic encryption (HE)** - a kind of encryption
where you can do math on the *encrypted* numbers, and the encrypted answer decrypts
to the correct result. So the server computes the match score on data it literally
cannot read.

1 + 2
(A + B)
1 + 2 = 3 (C)
(c)

HE -> 1, 2 -> A, B -> C (3) 

## 3. The two halves of the system

**1. A neural network (the "is this the same finger?" brain)**
A *Siamese network* turns a fingerprint image into a list of numbers (a "feature
vector") that captures the finger's identity. Same finger → similar vectors;
different fingers → different vectors. Matching becomes "are these two number-lists
close?"

(we get floating point numbers as the feature vectors 
convert into binary - so that it works for PSI)

**2. Homomorphic encryption (the "do it blindly" part)**
The client encrypts its feature vector and sends it to the server. The server does
the comparison math on the ciphertext and sends back an *encrypted score*. Only the
client (who holds the secret key) can decrypt it.

## 4. Who does what (the 3 containers)

| Container | Real-world role | What it does |
|-----------|----------------|--------------|
| **client** | The user's phone/device | Runs the neural net, generates keys, encrypts the fingerprint, and **is the only one that can decrypt**. The secret key never leaves here. |
| **main** | Coordinator server | Receives the encrypted query, fans it out to the worker(s), combines the encrypted results back. |
| **cluster1** | Matching worker | Holds the encrypted enrolled templates and does the actual encrypted comparison math. |

> The *paper* uses several cluster workers (it shards the user database across them
> to scale). This demo uses **one** cluster - same cryptography, just not sharded.
> That's a config change, not a redesign.

---

## 5. The architecture diagram (demo: 3 containers + 1 shared store)

```
                         ┌─────────────────────────────────────────┐
                         │           SHARED VOLUME                 │
                         │     (/workspace/shared_data)            │
                         │   keys/  models/  ciphertexts/  results/│
                         └─────────────────────────────────────────┘
                            ▲            ▲                 ▲
                            │            │                 │
              ┌─────────────┘            │                 └────────────┐
              │                          │                              │
     ┌────────┴─────────┐      ┌─────────┴────────┐          ┌──────────┴────────┐
     │     CLIENT       │      │      MAIN        │          │     CLUSTER1      │
     │  (user device)   │      │  (coordinator)   │          │  (match worker)   │
     │                  │ HTTP │                  │   HTTP   │                   │
     │ • neural net     │─────▶│ • receives query │─────────▶│ • encrypted match │
     │ • keygen 🔑      │      │ • calls worker(s)│◀─────────│ • on ciphertext   │
     │ • encrypt/decrypt│◀─────│ • returns result │   result │   only            │
     │ 🔒 SECRET KEY    │      │                  │          │                   │
     │   stays here     │      │ sees: ciphertext │          │ sees: ciphertext  │
     └──────────────────┘      └──────────────────┘          └───────────────────┘
        port 8888 (Jupyter)       port 8090                   (internal only)

   🔑 = holds the secret key    🔒 = only place that can decrypt
   Everything left of the client is BLIND - it only ever touches encrypted data.
```

---

## 6. Flow 1 - Enrollment (one-time setup)

```
  CLIENT
  ┌──────────────────────────────────────────────────┐
  │ 1. generate keys (public, secret, galois, relin) │
  │ 2. fingerprint image ──▶ neural net ──▶ feature  │
  │    vector (~25,000 numbers)                      │
  │ 3. ENCRYPT the vector  🔒                        │
  └───────────────────────┬──────────────────────────┘
                          │ write
                          ▼
              SHARED VOLUME: ciphertexts/  +  keys/(public,galois,relin)
                          │
                          └──▶ cluster1 later loads these as the
                               "enrolled templates" to match against
        (the SECRET key is written too but ONLY the client uses it)
```

## 7. Flow 2 - Authentication (the live request)

```
   CLIENT                         MAIN                      CLUSTER1
   (device)                    (coordinator)              (match worker)
     │                              │                          │
  [1]│ finger ─▶ net ─▶ vector      │                          │
     │ ENCRYPT it 🔒                │                          │
     │                              │                          │
  [2]│ ──── HTTP: encrypted query ─▶│                          │
     │                              │                          │
     │                           [3]│ ─── HTTP: forward ──────▶│
     │                              │                          │[4] ENCRYPTED MATCH
     │                              │                          │    • subtract query
     │                              │                          │      from each
     │                              │                          │      enrolled template
     │                              │                          │    • square it
     │                              │                          │    • dense projection
     │                              │                          │    ↳ all on ciphertext
     │                              │                          │      (server is blind)
     │                              │                       [5]│ ── encrypted score ──▶
     │                              │◀─────────────────────────┘   (~0.23 s)
     │                           [6]│ aggregate result
     │◀──── HTTP: encrypted score ──┤
     │                              │
  [7]│ DECRYPT 🔓 + final sigmoid   │
     │ score 0.998 > 0.99 ?         │
     │ ✅ AUTHENTICATED             │
```
A1 A2 A3 .. A512 
A -> C1 C2 C3 .. C512 -> C (>0.99)
CKKS Scheme - SEAL

The 7 beats to narrate:
1. **Client encodes & encrypts** - fingerprint → feature vector → ciphertext. Secret key stays on device.
2. **Send** - only ciphertext goes over the network.
3. **Coordinator forwards** - `main` fans out to worker(s).
4. **Blind matching** - `cluster1` computes on encrypted data: *subtract → square → project*. Never decrypts.
5. **Encrypted score returned** - still ciphertext, ~0.23 s.
6. **Aggregate** - `main` combines results (with multiple clusters, merges them here).
7. **Client decrypts & decides** - final step + 0.99 threshold → Authenticated / Rejected.

**The line to land:** *"Everything from the network onward is blind - `main` and
`cluster1` only ever touch ciphertext. Plaintext biometrics and the secret key
exist only on the client. A fully compromised server leaks nothing usable."*

---

## 8. What exactly are the "enrolled templates"?

An enrolled template is **a registered user's fingerprint after the neural net
compresses it into a small list of numbers, then encrypted.** It is never the raw
image.

### How one template is made (from `Blind-Touch-Enroll.ipynb`)

```
raw fingerprint image  (224×224 grayscale)
        │  feature_model.predict()
        ▼
feature map 7×7×512  ──Flatten──▶  25,088-dim feature vector
        │  UnitNormalization()      (unit length, so scale doesn't matter)
        ▼
np.matmul(vector, weight)       ← weight = the matching layer's kernel (25088 → 16)
        ▼
  16 numbers   ← THIS is one finger's compact "template"
        │  CKKS encode + encrypt 🔒
        ▼
  ciphertext  → saved as ctxt1 / ctxt2 / ctxt3 on the shared volume
```

So one enrolled template is just **16 numbers** - a compressed fingerprint
signature - and then it's **encrypted**.

### The elegant packing trick

```python
result1 = feature_model.predict(test_data[:512])      # 512 fingerprints
result1 = np.matmul(normalized_features, weight)        # → 512 × 16 numbers
ctxt1 = encryptor.encrypt(ckks_encoder.encode(result1.flatten(), scale))
```

- 512 fingerprints × 16 numbers each = **8,192 numbers**
- CKKS has exactly **8,192 slots** per ciphertext

So **`ctxt1` packs 512 enrolled users into a single ciphertext.** That's why one
homomorphic operation can match the query against all 512 at once - and why it's
fast.

### "Enrolled template" = three things stacked

1. **A biometric template** - a math representation used for comparison, never the raw image (here, the net's feature vector projected down to 16 numbers).
2. **Pre-computed** - the heavy neural-net work is done at enrollment, baked into the stored template, so matching later is cheap.
3. **Homomorphically encrypted** - the stored form (`ctxt1`) is ciphertext; the cluster holds it but has no secret key, so it can never read who's enrolled.

### Used at match time (`server/cluster1/service.py`)

```python
sub_ctxt = evaluator.sub(ctxt, clustering_ctxt)   # query − enrolled template
result   = fc1_layer(evaluator, square(...))       # square the difference, project
```

Subtract query from enrolled template, square it → a **distance measure**. Small
difference → same finger → high score. All on ciphertext.

**One-liner:** *"An enrolled template is a registered fingerprint compressed by the
neural network into 16 numbers and then homomorphically encrypted. Hundreds are
packed into one ciphertext, so the server matches against all of them at once -
blindly."*

---

## 9. What is the role of `main`, if it just passes things to and fro?

In **this demo** (one cluster), `main` *is* basically a passthrough - the client
could talk to `cluster1` directly. But `main` exists because of what the **full**
system looks like: a **scatter-gather** coordinator.

### The real (multi-cluster) picture

The paper shards the enrolled database **across many clusters** - cluster1 holds
users 1-512, cluster2 holds 513-1024, etc. No single cluster has everyone.

```
                          ┌──▶ cluster1  (users 1-512)    ─┐
   client ──query──▶ MAIN ─┼──▶ cluster2 (users 513-1024) ─┤  partial
                          └──▶ cluster3  (users 1025-...)  ─┘  encrypted
                                                               results
                              MAIN combines them ◀────────────┘
                              into ONE ciphertext
   client ◀── single result ──┘
```

`main` does two things the client can't easily do itself:

1. **Fan-out (scatter).** Sends the query to *all* clusters in parallel
   (`fetch_all` / `asyncio.gather` in `main/service.py`). The client never needs to
   know how many clusters exist.
2. **Aggregate (gather + compress).** Each cluster returns a *partial* encrypted
   result; `main` merges them into a **single** ciphertext. From the code:
   ```python
   # Single cluster: its partial result IS the compressed result. (With 3
   # clusters this would be evaluator.add_many([ctxt1, ctxt2, ctxt3]).)
   ```
   It works cleanly because each cluster rotates its result into a *different slot*
   (`rotate_vector(..., -(CLUSTER_NUM-1), ...)`), so adding them all merges each
   cluster's scores into their own slots in one combined score-vector. The client
   then decrypts **one** ciphertext instead of N.

### The three real reasons `main` exists

| Role | Why it matters |
|------|---------------|
| **Single entry point** | The client knows *one* address. Clusters can scale behind `main` without the client changing. |
| **Scatter** | Broadcasts the query to all shards in parallel. |
| **Gather + compress** | Combines many partial encrypted results into one ciphertext, so the client does a single decryption. |

**How to say it:** *"`main` is the coordinator in a scatter-gather design. Because
the enrolled database is sharded across many cluster workers, `main` broadcasts the
encrypted query to all of them and homomorphically combines their partial results
into one ciphertext for the client. In my single-cluster demo there's only one
worker, so `main` is effectively a passthrough - but it's the component that makes
the system scale to millions of users across many servers, which is the paper's
whole point."*

---

## 10. Where everything lives in the code

- `client/*.ipynb` - **Client** (feature extraction), **Enroll** (keygen + encrypt templates), **Auth** (encrypt query, send, decrypt, decide).
- `server/cluster1/service.py` - the heart: encrypted match (`evaluator.sub` → `square` → `fc1_layer`), ~80 lines.
- `server/main/service.py` - coordinator; scatter (`fetch_all`) + gather (`add_many`).
- `docker-compose.yml` - wires the 3 containers + the one shared volume (stands in for the paper's networked storage).
