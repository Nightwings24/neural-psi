# The CNN Feature Extractor, From Scratch (Stage 0 + Stage 1)

*A self-contained explanation of how a raw fingerprint image becomes 16 numbers — the
"what, why, and how" of every step. Written so you can teach it to anyone, assuming no
background beyond basic algebra. This covers **Stage 0 (the raw input)** and **Stage 1 (the
CNN)** of Neural-PSI; Stage 2 (binarization) and Stage 3 (the PSI match) are separate documents.*

---

## Table of contents
0. [The one-sentence goal](#0-the-one-sentence-goal)
1. [Stage 0 — the raw fingerprint input](#stage-0--the-raw-fingerprint-input)
2. [Stage 1 — the CNN, block by block](#stage-1--the-cnn-block-by-block)
3. [How the CNN is trained (the part that makes it work)](#3-how-the-cnn-is-trained-the-part-that-makes-it-work)
4. [Why every design choice was made](#4-why-every-design-choice-was-made)
5. [The 2-minute version (a cheat sheet)](#5-the-2-minute-version)

---

## 0. The one-sentence goal

> **Turn a fingerprint image (thousands of pixels) into just 16 numbers that capture the
> *identity* of the finger** — so that two scans of the *same* finger give *similar* 16
> numbers, and two *different* fingers give *different* 16 numbers.

Those 16 numbers are called the **embedding** (or *feature vector*). Everything downstream
(the binary code, the private match) is built on top of them. The whole job of Stage 1 is to
produce a *good* embedding.

The full pipeline, for context:

```
fingerprint image  ──Stage 1: CNN──▶  16 floats (e)  ──Stage 2──▶  128 bits  ──Stage 3: PSI──▶  match
   (this doc)                          (this doc)
```

---

## Stage 0 — the raw fingerprint input

### 0.1 What a fingerprint image actually *is*

**What.** A fingerprint image is just a **grid of numbers**. Each cell of the grid is a
**pixel**, and each pixel holds one **brightness value** from 0 to 255:

- **0 = black** → a ridge (the raised lines of the fingerprint, where ink/contact is darkest).
- **255 = white** → a valley or background (the gaps between ridges).

There is no colour — it is **grayscale**. So a `96×96` image is literally a table of 96×96 =
9,216 numbers between 0 and 255. A `224×224` image is 50,176 numbers.

**Why this matters.** The computer never "sees" a fingerprint the way we do. It only ever sees
this grid of numbers. Everything the network does is arithmetic on this grid.

### 0.2 The dataset — SOCOFing

We train and test on **SOCOFing** (Sokoto Coventry Fingerprint dataset), a public set of *real*
fingerprints:

- **6,000 images** from **600 people**, **10 fingers each** (600 × 10 = 6,000).
- Each finger has a **Real** capture (a clean scan — used to *enroll* / register the finger)
  and several **Altered** captures (deliberately distorted with synthetic *obliteration*,
  *rotation*, or *z-cut* damage — used as the *probe* at authentication time, to simulate a
  messy real-world re-scan).
- **Split:** 4,800 fingers are used to **train** the network; **1,200 fingers are held out**
  and used *only* to measure accuracy. The held-out fingers are never seen during training, so
  the accuracy numbers are honest (no "studying the answer key").

> **Honesty note.** Our accuracy numbers use the **Altered-Easy** captures — the mildest
> distortion level. It is an intentionally *easy*, single-dataset benchmark. A different
> sensor or a different day (cross-sensor / cross-session) would be harder.

### 0.3 The only preprocessing — divide by 255

**What.** Before the image enters the network, every pixel is rescaled from `[0, 255]` to
`[0, 1]`:

```
x_norm = x_pixel / 255
```

**Why.** Neural networks train far more stably when their inputs are small numbers around 0–1
rather than 0–255. Large inputs make the internal arithmetic swing wildly and the training
diverge. This is the *only* preprocessing — **no alignment, no ridge-enhancement, no
segmentation**. The network is trained to cope with raw, messy scans directly.

### 0.4 Data augmentation — teaching the network to be forgiving

**The problem.** Two scans of the *same* finger are **never** pixel-identical. You press a
little harder, your finger is rotated 5°, it's slightly wetter. If the network demanded an
exact pixel match, it would reject you constantly.

**What we do (the how).** During training, every time we show the network a finger, we first
apply **random distortions** to it — simulating a "different scan of the same finger":

- **Rotation:** a random angle between **−15° and +15°**.
- **Translation (shift):** up to **6% of the image width** in each direction (≈6 px at 96×96).
- **Gaussian noise:** add small random speckle (standard deviation σ = 6 on the 0–255 scale),
  then clip back into range.

**How training pairs are built:**
- A **genuine pair** = **two independently-distorted copies of the *same* Real print**
  (label = 1, "same finger").
- An **impostor pair** = distorted copies of **two *different* fingers** (label = 0,
  "different fingers").

**Why this is the whole trick.** By showing the network thousands of "same finger, but shifted
/ rotated / noisy" pairs, we *force* it to learn what stays the same about a finger's identity
when the scan conditions change. That learned invariance is exactly what lets it recognise your
finger later from a fresh, imperfect scan.

---

## Stage 1 — the CNN, block by block

### 1.1 Why a CNN at all (and not "classical" fingerprint matching)

**The classical way.** Traditional fingerprint systems extract **minutiae** — the specific
points where ridges end or split — using hand-written image-processing rules. This is
**brittle**: it needs high-quality images, breaks under noise, and produces a *variable-length*
list of points that is slow and awkward to compare across a big database.

**The CNN way.** A **Convolutional Neural Network** *learns from data* which visual patterns
signal identity, and squeezes **any** fingerprint — good or bad quality — into a **single,
fixed-length vector** (here, 16 numbers).

**Why "fixed-length" is the key word.** Both privacy backends we care about — Homomorphic
Encryption (the original Blind-Touch) and our PSI matcher — require every user's template to be
the *same, constant size*. A variable-length minutiae list can't be used; a fixed 16-number
vector can. The CNN is what makes the whole privacy-preserving pipeline possible.

### 1.2 The architecture at a glance

The network is **5 convolutional blocks**, then a **flatten**, then **one small layer that
outputs 16 numbers**:

| Stage | Operation | Output shape (96×96 input) | What it roughly captures |
|---|---|---|---|
| Input | — | 1 × 96 × 96 | raw pixels |
| Block 1 | Conv(3×3, 32) + BN + SiLU + MaxPool(2) | 32 × 48 × 48 | edges, ridge boundaries |
| Block 2 | Conv(3×3, 64) + BN + SiLU + MaxPool(2) | 64 × 24 × 24 | ridge flow |
| Block 3 | Conv(3×3, 128) + BN + SiLU + MaxPool(2) | 128 × 12 × 12 | minutiae-like patterns |
| Block 4 | Conv(3×3, 256) + BN + SiLU + MaxPool(2) | 256 × 6 × 6 | finger-region shapes |
| Block 5 | Conv(3×3, 512) + BN + SiLU + MaxPool(2) | 512 × 3 × 3 | abstract "identity code" |
| Flatten | — | 4,608 | lay everything in a row |
| **FC-16** | Linear(4608 → 16) | **16** | **the feature vector `e`** |

Notice the **two opposite trends**: the picture keeps **shrinking** (96→48→24→12→6→3) while the
number of **channels keeps growing** (1→32→64→128→256→512). This is the core idea of a CNN:
**trade "where exactly is it" for "what pattern is it."** Early layers know precise locations of
tiny edges; late layers know abstract "this looks like this identity" but have thrown away exact
position.

> The table above is for the **96×96** teaching model. The **production model uses 224×224**
> input, so the picture shrinks 224→112→56→28→14→7 and flattens to 512×7×7 = **25,088** numbers
> before the FC-16 layer. Everything else is identical.

Now let's open up each operation inside a block.

### 1.3 The convolution — the "pattern detector"

**What.** A convolution slides a tiny **filter** (here a 3×3 grid of learned numbers) across the
image. At each position, it multiplies the filter's 9 numbers by the 9 pixels underneath, adds
them up, and writes the single result into an output grid:

```
Y(i,j) = Σ (filter values × the 3×3 patch of pixels around position (i,j)) + bias
```

**How to picture it.** Think of the filter as a small **stencil** or a **flashlight** that
scans the whole image looking for one specific little pattern — say, "a diagonal edge." Where
the image matches the stencil, the output is large (bright); where it doesn't, the output is
small (dark). The output grid is called a **feature map**: a map of *where* that pattern occurs.

**Why 32 filters in Block 1 (and 64, 128, … later)?** One filter finds one pattern. To detect
*many* patterns (edges at different angles, ridge segments, dots), you use **many filters** — 32
of them in Block 1 — each producing its own feature map. That's why the output has 32 "channels."

**Where do the filter numbers come from?** They are **not** hand-designed. They start random and
are **learned during training** (Section 3) so that, by the end, each filter has automatically
become a detector for some pattern that helps tell fingers apart.

*("Same" padding: the image is invisibly bordered with a rim so the output keeps the same width
and height as the input — the shrinking happens only at the pooling step, not here.)*

### 1.4 Batch normalization — keeping the numbers sane

**What.** After each convolution, we **normalize** each channel's outputs: subtract their mean
and divide by their spread (standard deviation), then rescale with two *learned* knobs:

```
normalized = (Y − mean) / sqrt(variance + tiny_ε)
output     = γ × normalized + β        (γ, β are learned per channel)
```

**Why.** As data flows through 5 stacked blocks, the numbers can drift to huge or tiny scales,
which makes training unstable or painfully slow. Batch norm **resets each channel to a clean,
standard scale** at every block, keeping the whole deep stack trainable. The learned `γ, β` let
the network undo the normalization if a particular channel actually needs a different scale — so
it loses no expressive power.

**Analogy.** It's like re-centring and re-scaling a noisy measurement back to a standard range
after every step, so errors don't snowball.

### 1.5 The activation (SiLU) — adding the "bend"

**Why we need *any* activation.** A convolution (and the FC layer) is a **linear** operation —
just weighted sums. If you stack linear operations, the result is *still* just one big linear
operation, no matter how many layers. Linear maps can only draw straight-line boundaries — they
can't model the complicated, curved patterns that distinguish fingerprints. So between layers we
insert a **non-linear** function; this is what lets a deep network represent complex shapes.

**What SiLU is (the how).**

```
SiLU(x) = x × sigmoid(x) = x / (1 + e^(−x))
```

For large positive `x` it acts like the identity (passes the value through); for large negative
`x` it gently squashes toward 0; near 0 it's a smooth curve.

**Why SiLU instead of the more common ReLU?** ReLU (`max(0, x)`) hard-clips every negative value
to exactly 0, which can "kill" neurons and has a sharp corner. SiLU is **smooth everywhere** and
lets a **small negative signal leak through**, which improves the flow of learning signals
(gradients) down through 5 stacked blocks — the Blind-Touch authors found it converges more
reliably than ReLU on fingerprint data.

### 1.6 Max pooling — shrink, and gain robustness

**What.** MaxPool(2) looks at each non-overlapping **2×2 square** of the feature map and keeps
**only the largest value**, throwing away the other three. This **halves** the width and height
(96→48→24→12→6→3).

**Why (two reasons).**
1. **Efficiency:** fewer numbers to process in later layers.
2. **Invariance (the important one):** by keeping only "was this pattern *present* in this little
   region" and discarding its *exact* pixel position, the network becomes tolerant to small
   shifts and rotations. This is the architectural partner to the *augmentation* from Stage 0:
   augmentation *teaches* shift-tolerance, pooling *builds it into the structure*. Together they
   make the embedding stable across re-scans of the same finger.

### 1.7 One block, and the pattern repeated 5 times

Each block is the same four steps — **Conv → BatchNorm → SiLU → MaxPool** — and we stack **five**
of them. With each block, the network sees a *larger* effective region of the original image and
builds *more abstract* features:

edges (Block 1) → ridge flow (Block 2) → minutiae-like structure (Block 3) → whole finger-region
shapes (Block 4) → an abstract "identity code" (Block 5).

*(These labels are interpretive — the network is never told "find minutiae." It discovers
whatever patterns happen to help, and they tend to grow more abstract with depth.)*

### 1.8 Flatten + FC-16 — down to 16 numbers

**What.** After Block 5 we have 512 channels each of size 3×3 (for 96×96 input) — that's
512 × 3 × 3 = **4,608** numbers. We lay them all out in a single long row (**flatten**), then
pass that row through **one final linear layer** that outputs exactly **16 numbers**:

```
e = W · x_flat + b        (W is 4608×16 for the 96px model, 25088×16 for the 224px model)
```

Those **16 numbers, `e`, are the feature vector** — the compact fingerprint of the fingerprint.
This is the output of Stage 1 and the entire input to Stage 2.

**Two things to know about `e`:**
- The values are **raw** (not normalized to any fixed range) — a real one looks like
  `e = [−7.41, −33.85, +16.92, −2.61, −4.30, −48.06, …]`, i.e. ordinary signed numbers that can
  be fairly large.
- It is computed **entirely on the client device, in plaintext**. The 16 floats **never leave
  the client** — Stage 2 converts them to a binary code first, and the server only ever sees a
  privacy-masked version of *that*. The server never receives the raw embedding or the image.

**Why exactly 16?** See Section 4 — it's chosen to match the encryption backend's packing, not
arbitrary.

---

## 3. How the CNN is trained (the part that makes it work)

Everything above describes the *machinery*. But a freshly-built network has **random** filter
values and produces meaningless embeddings. **Training** is what turns it into an identity
detector. This is the most important "why."

### 3.1 Why not just build a classifier?

A natural first idea: "train it to output *which* of the 600 people this is." **This does not
work for authentication**, because authentication is **open-set**: new users enrol *after* the
network is trained. A fixed 600-way classifier can't recognise person #601. (There's also a
capacity reason: you can't cram thousands of distinct identity "slots" into a mere 16-number
space — attempts to do so collapse.)

**The fix: learn a *similarity*, not a label.** We train the network so that its embedding space
has one property: **same finger → nearby vectors; different fingers → far-apart vectors.** Then
enrolling a brand-new finger is trivial — just store its embedding — and matching is just
"is the new embedding close to a stored one?" This is called **metric learning**.

### 3.2 The Siamese setup

To *teach* "same = close," we compare **pairs**. We run **two copies of the exact same feature
extractor** (they share identical weights — hence "**Siamese**," like twins) on two images:

```
e1 = features(image1)
e2 = features(image2)
```

Then a tiny **comparison head** decides if they're the same finger:

```
d = |e1 − e2|              (element-wise absolute difference — a 16-number "disagreement" vector)
score = w1 · d + b1        (one linear unit collapses those 16 into a single number)
p = sigmoid(score)         (squash to a probability between 0 and 1: "are these the same finger?")
```

### 3.3 The loss — the "grade" that drives learning

We show the network many labelled pairs (`y = 1` genuine, `y = 0` impostor) and score it with
**binary cross-entropy (BCE)**:

```
Loss = −[ y·log(p) + (1−y)·log(1−p) ]   averaged over all pairs
```

In words: the loss is **small** when the network is confident *and correct* (says ≈1 for genuine,
≈0 for impostor), and **large** when it's confidently wrong. Training repeatedly nudges all the
filter values (via **gradient descent**) to make this loss smaller.

**What this actually forces.** To get a low loss, the network *must* make genuine embeddings
land close together (so `|e1 − e2|` is small) and impostor embeddings land far apart. That is
precisely the "same = close" property we wanted — achieved *without ever hand-labelling what the
16 numbers mean*. The meaning emerges automatically from the pressure of the loss.

### 3.4 After training: keep the extractor, throw away the head

Once trained, we **keep only the feature extractor** (the part that turns an image into `e`) and
**discard the comparison head** (`w1, b1`). Why? Because Stage 2 and Stage 3 do the matching a
completely different way — by **Hamming distance on a binary code**, privately — not with this
learned head. The head was just a training aid to shape the embedding space; its job is done.

---

## 4. Why every design choice was made

| Choice | Why |
|---|---|
| **5 convolutional blocks** | Blind-Touch's own depth ablation found 5 is the sweet spot: F1 = 93.8% at 5 blocks vs 92.4% at 4 (under-fits) and 89.3% at 6 (over-fits on a dataset this size). *(These are the original paper's numbers, not re-measured by us.)* |
| **16-dimensional output** | Not arbitrary. The original encryption backend packs 16 values per user into ciphertext "slots"; with 8,192 slots that's exactly 512 users per ciphertext. We keep 16 so the *same* trained CNN can feed **either** backend (encryption or PSI) — only the step *after* the CNN changes. |
| **SiLU over ReLU** | Smooth, and lets small negative signals through → better gradient flow and more reliable convergence through 5 stacked blocks. |
| **Max pooling + augmentation** | Together they make the embedding tolerant to the shifts/rotations/noise of a real re-scan (structure + training data, respectively). |
| **Resolution: 96×96 vs 224×224** | This matters more than anything else. Training the small CPU model (96×96, 6 epochs) gives a **2.50%** error floor; the full model (224×224, 150 epochs, the original Blind-Touch spec) gives **0.33%** — a **7.6× improvement** from resolution + training alone. |

**The single most important finding.** The dominant source of error in the whole system is the
**embedding quality**, not the later binarization or matching. Improving the CNN (resolution,
training) moved the floor 7.6×; the binarization only adds ~1.5 points on top. So the CNN is the
part that most rewards further investment — e.g. a wider embedding than 16 dimensions.

---

## 5. The 2-minute version

> A fingerprint image is just a grid of brightness numbers. A **CNN** slides small learned
> **filters** over it to detect patterns (edges → ridges → identity), using **5 blocks** that
> each **shrink the picture but grow the number of pattern-channels** — trading "where exactly"
> for "what pattern." Between steps, **batch norm** keeps the numbers on a sane scale and **SiLU**
> adds the non-linear "bend" that lets it learn complex shapes; **max pooling** shrinks and adds
> tolerance to small shifts. At the end, everything is flattened and squeezed by one layer into
> **16 numbers** — the *embedding*.
>
> The magic is in the **training**: because new users enrol after training, we don't build a
> classifier — we use a **Siamese** setup that learns a *similarity*, pushing **same-finger
> embeddings close together and different-finger embeddings far apart** (via a binary
> cross-entropy loss on genuine/impostor pairs). **Data augmentation** (random rotate/shift/noise
> on the same print) teaches it to shrug off the differences between two real scans of the same
> finger.
>
> The result: two scans of your finger → two nearby 16-number vectors; someone else's finger →
> a far-away one. Those 16 numbers are the handoff to Stage 2, which turns them into a 128-bit
> code for private matching.

---

*Accuracy note: this document matches the actual implemented model (`implementation/week1/model.py`,
`train_gpu_224.py`) — a Siamese network with an `|e1−e2| → Linear(→1)` head trained by BCE, whose
feature extractor outputs a raw (un-normalized) 16-D embedding. For the downstream stages see the
Stage-2 (Super-Bit) and Stage-3 (PSI) writeups.*
