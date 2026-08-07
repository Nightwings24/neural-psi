# Neural-PSI — Final Presentation Script

**Total: ~21 minutes** (target window 20–23 min, includes the ~4 min live demo)

| Speaker | Slides | Topic | Time |
|---|---|---|---|
| **Ayan** | 1–10 | Intro, problem, prior work, our solution, architecture | ~7 min |
| **Prabhu** | 11–15 | Feature extraction & quantization | ~4 min |
| **Rineet** | 16–24 | Cryptography: the FLPSI engine | ~6 min |
| **Prabhu** | 25 | Live demonstration | ~4 min |
| **Prabhu** | 26–29 | Results, conclusion & future work | ~2.5 min |

> **Handover rule:** whoever finishes says the next person's name — "…and Prabhu will take you through how we get there." Never let a slide change happen in silence.

---

# PART 1 — AYAN (Slides 1–10, ~7 min)

## Slide 1 — Title (~40 s)

> "Good morning. I'm Ayan, and with me are Rineet and Prabhu. This is our internship project at QNu Labs — **Privacy-Preserving Fingerprint Authentication** — carried out under Dr. Amit Kumar Chauhan, Ayan Chattopadhyay, and Dr. Rohitkumar Upadhyay.
>
> I'll cover the problem and our proposed solution. Prabhu will take the neural network and the quantization bridge, Rineet will cover the cryptography, and then Prabhu will run a live demo and close with our results."

**[NEXT SLIDE]**

## Slide 2 — Outline (~20 s)

> "Briefly, the structure: we start with why fingerprint privacy is hard, then walk the pipeline in three stages — a CNN that extracts features, a quantizer that turns them into binary, and a private set intersection protocol that does the matching. Then results and a demo."

**[NEXT SLIDE]**

## Slide 3 — What is a Fingerprint? (~40 s)

> "Start with the biology. Fingerprint ridges form before birth and never change. They're **unique** — even identical twins differ. They're **immutable** — the pattern persists for life. And they're **universal**.
>
> Those three properties are exactly what make a fingerprint excellent identification. But notice the flip side, in red: **irreplaceable if leaked**. That single word is the entire motivation for this project."

**[NEXT SLIDE]**

## Slide 4 — How Authentication Works (~40 s)

> "Here's the standard pipeline, and it has two phases. **Enrollment**: capture your finger, extract features, store a template. **Authentication**: capture again, extract features, compare against the stored template.
>
> Capture, extract, compare, accept or reject. If the similarity score beats a threshold, you're genuine.
>
> Everything we've built lives in those middle two boxes. 'Extract features' becomes a 16-dimensional neural embedding, and 'compare' becomes a fuzzy private set intersection — done without the server ever seeing your fingerprint."

**[NEXT SLIDE]**

## Slide 5 — The Problem (~50 s)

> "So why is this hard? Two reasons.
>
> First: **a biometric is the worst possible password**. If a password leaks, you rotate it. If a cryptographic key leaks, you regenerate it. If your fingerprint leaks — you have nine left, and then you're done. Forever compromised.
>
> Second, and more subtle: the **trusted-third-party problem**. In every legacy system, some matching service holds your biometric in the clear at comparison time. You're not trusting the maths, you're trusting the operator.
>
> What we actually want is the expression at the bottom: the server should learn *whether* there was a match — and nothing about the database or the query beyond that."

**[NEXT SLIDE]**

## Slide 6 — Current Cryptographic Tools (~60 s)

> "This is the landscape of what already exists, and every family fails on at least one axis.
>
> **One-way hashing** is the obvious first idea — store a digest, compare digests. It fails immediately: hashing is exact-match, and two scans of your finger are never bit-identical. One flipped bit and the hash is unrecognisable.
>
> **Fuzzy commitment** and **fuzzy vault** were designed to fix exactly that — they tolerate noise using error-correcting codes or by hiding real minutiae among fake 'chaff' points. But fuzzy vault is vulnerable to correlation attacks and isn't revocable.
>
> **Fuzzy extractors** derive a stable key plus public helper data, but they have high error rates on real biometrics.
>
> **Cancelable biometrics and BioHashing** apply a keyed non-invertible transform — but accuracy drops, and security rests entirely on keeping the token secret.
>
> Then **homomorphic encryption** — that's Blind-Touch, the paper we build on. It's accurate, but expensive.
>
> And the bottom row, **FLPSI** — fuzzy labelled private set intersection — is what we use. Its only requirement is that inputs are **binary codes compared by Hamming distance**. Remember that constraint; it's the whole reason the middle of our pipeline exists."

**[NEXT SLIDE]**

## Slide 7 — Blind-Touch (~60 s)

> "Blind-Touch is the closest prior work, from AAAI 2024, and it's worth understanding properly because we reuse half of it.
>
> A Siamese CNN maps a fingerprint to a 16-dimensional embedding, and then the matching runs **under CKKS homomorphic encryption** — the server computes on ciphertext and never decrypts. That's the identity on screen.
>
> And we keep their feature extractor. **The accuracy comes from the CNN, not from the encryption** — that's the key observation.
>
> But the encryption layer costs them three things. A **117-megabyte Galois rotation key** on every server. A **packing structure** — 8,192 slots divided by 16 values is 512 users per ciphertext, so cost grows in blocks of 512, and they only ever evaluate at 5,000 users. And **no labels** — it returns a match score, never *which* record matched.
>
> So the matching layer is the bottleneck. That's the piece we replace."

**[NEXT SLIDE]**

## Slide 8 — Our Proposed Solution (~20 s)

> "Which brings us to our system: **Neural PSI**."

*(Pause. Let it land — this is the pivot of the talk.)*

**[NEXT SLIDE]**

## Slide 9 — What is PSI? (~50 s)

> "Private set intersection: two parties, Alice with a set X, Bob with a set Y. They learn the intersection — and provably nothing else.
>
> Why is that the right tool for biometrics? Because it's **asymmetric**. Alice learns the label of her match; Bob, the server, learns nothing at all.
>
> But there's a catch, and it's the same catch as one-way hashing: standard PSI tests **exact** set membership, and biometric codes are noisy.
>
> That's why we need the *fuzzy* variant — FLPSI — which accepts when *t* out of *T* sub-samples match rather than demanding an exact hit. Rineet will unpack how that works."

**[NEXT SLIDE]**

## Slide 10 — System Architecture (~60 s)

> "Here's the whole system in one picture — and the most important thing is the vertical split.
>
> On the **left**, the client device, entirely in plaintext. Your fingerprint is captured, the CNN produces a 16-D embedding, and the Super-Bit quantizer turns that into a 128-bit binary code. **All of that happens on your own device.**
>
> On the **right**, the untrusted server, which runs the FLPSI engine — masked OPRF, threshold PSI, Ring-OLE, Shamir secret sharing — over a template database holding roughly **16 bytes per user**.
>
> Enrollment, on the far right, is a one-time upload of that 128-bit code.
>
> And the line at the bottom is the guarantee: **Bob never learns Alice's fingerprint, her embedding, or her query code — not even whether a match occurred. Only Alice learns the result.**
>
> Now Prabhu will take you through the two client-side stages."

**[HANDOVER → PRABHU] [NEXT SLIDE]**

---

# PART 2 — PRABHU (Slides 11–15, ~4 min)

## Slide 11 — Preprocessing & Convolution (~55 s)

> "Thanks Ayan. I'll cover how we get from an image to a binary code.
>
> The input is a 224×224 grayscale fingerprint, normalized to zero-to-one. And notice what we **don't** do: no alignment, no filtering, no minutiae extraction. Classical fingerprint systems spend enormous effort there. We hand the network raw pixels and let it learn.
>
> Because two scans of the same finger are never identical, we augment during training — random rotation up to fifteen degrees, translation, and Gaussian noise. That's what teaches the network to be *invariant* to how you happened to press your finger.
>
> The network is five convolutional blocks: convolution, batch-norm, SiLU activation, max-pool. And watch the two opposite trends — channels **grow** 32 to 512, while the spatial map **halves** each block, 224 down to 7. That's the core idea of a CNN: it trades 'where exactly' for 'what pattern'. Early layers see edges; the last layer holds an abstract identity code."

**[NEXT SLIDE]**

## Slide 12 — Feature Vector & Siamese Training (~55 s)

> "After the fifth block we have 512 channels of 7×7 — that's **25,088 numbers**. One linear layer compresses that all the way down to **16**.
>
> Those 16 numbers are the feature vector, and they're computed **entirely client-side**. The server never sees them.
>
> Now, how do we train this? Not as a classifier — because new users enrol *after* training, so a fixed-class classifier can't recognise person six-hundred-and-one.
>
> Instead we use a **Siamese** setup: two identical copies of the network process two fingerprints, and the loss on screen pushes embeddings of the **same** finger close together and **different** fingers far apart. That's metric learning — we learn a *similarity*, not a label. And that's what makes enrollment trivial later: just store the embedding."

**[NEXT SLIDE]**

## Slide 13 — Binarization Comparison (~55 s)

> "Now the bridge. The CNN gives us 16 floating-point numbers; FLPSI needs **binary strings compared by Hamming distance**. Something has to convert. We evaluated three options.
>
> **Method one, naive sign** — threshold each float at zero. Two problems: a float near zero flips between scans purely from noise, and 16 bits gives only 65,536 possible codes, so codes collide. Ten percent error rate.
>
> **Method two, ITQ** — learn a rotation that aligns the axes with the data before thresholding. It genuinely fixes the borderline-flip problem and roughly halves the error. But it's **still 16 bits**. The capacity ceiling is untouched.
>
> **Method three, Super-Bit** — this is what we chose. Instead of 16 bits we expand to **128**, using orthonormalised random projections. **1.88% equal error rate** on the 224 model — better than ITQ by a factor of three, and within about one and a half points of the floating-point ceiling.
>
> Bottom line: only Super-Bit has both enough code space *and* the statistical properties the protocol needs."

**[NEXT SLIDE]**

## Slide 14 — The Three Quantization Stages (~60 s)

> "So how does Super-Bit actually work? Three stages.
>
> **Stage one, centering.** Neural embeddings don't sit neatly around the origin — they cluster in a few directions. If we sliced that with planes through the origin, the bits would be heavily lopsided. So we subtract the population mean, computed offline.
>
> **Stage two, the Super-Bit projection** — this is the heart of it. The intuition: think of each embedding as an **arrow**. Each bit asks one question — *which side of a random dividing plane is this arrow on?* Similar arrows land on the same side of most planes; different arrows get split about half the time.
>
> The 'Super-Bit' refinement is that the planes are made **orthonormal in blocks** rather than purely random. Purely random planes sometimes come out nearly parallel by accident, which wastes bits on redundant measurements. Forcing them perpendicular means every bit measures something fresh — a **minimum-variance** estimate of the angle, for free.
>
> **Stage three, thresholding at the median** rather than zero — because the protocol requires every bit to be a genuine 50/50 coin flip.
>
> And at the bottom — honesty. We evaluated a fourth step, metric whitening. It **helps a weak model but hurts a well-trained one**, so we measured it, and dropped it."

**[NEXT SLIDE]**

## Slide 15 — Chosen Method (~25 s)

> "To summarise the bridge in two equations: the projection matrix is eight orthonormal blocks stacked, and each bit is a threshold comparison. Two prints of the same finger differ in about **8 bits out of 128**; two different fingers differ in about **64** — half the code. That gap is what the protocol tests.
>
> Rineet will now explain how you test that gap without either side revealing anything."

**[HANDOVER → RINEET] [NEXT SLIDE]**

---

# PART 3 — RINEET (Slides 16–24, ~6 min)

## Slide 16 — The Problem (~45 s)

> "Thanks Prabhu. My job is the cryptography — how we compare two codes privately.
>
> And the core difficulty is on the left: **two templates are never exactly equal**. Noise, rotation, pressure — a few bits always shift. What we actually need is the line in the box: *if the Hamming distance is within a threshold, reveal the label.*
>
> Ordinary PSI can't do that. It only tests exact equality. It has no notion of 'close enough'.
>
> FLPSI does, and with a precise leakage profile: the **server learns nothing**, and the **client learns the label on a match — plus which sub-samples collided**. We state that second part explicitly because it's the honest, complete leakage profile, not just the headline."

**[NEXT SLIDE]**

## Slide 17 — Cryptography Stack (~45 s)

> "Here's the whole engine at a glance, and it's worth seeing the shape before the details.
>
> The **client** takes her 128-bit code, generates T sub-sampling masks, and pushes them through a masked OPRF to get a set X-prime.
>
> The **server** takes its database and labels, splits each label into Shamir shares, and feeds those into Vector Ring-OLE and threshold PSI.
>
> Those two halves meet in the middle, and the rule is at the bottom: **if t of the T sub-samples match, the client can reconstruct the label.** Otherwise she gets nothing.
>
> I'll now walk the three primitives that make this possible."

**[NEXT SLIDE]**

## Slide 18 — Primitive 1: mOPRF (~50 s)

> "First primitive: the **masked oblivious PRF**.
>
> The problem it solves: we sub-sample the code — take a random subset of bit positions — and compare those. But if we compared the raw sub-samples, then every time there's a match, the server learns those actual biometric bits. Enough matches and it reconstructs the whole template.
>
> So instead we push each sub-sample through a PRF whose key only the server holds — but obliviously, using a garbled circuit. The client gets pseudorandom outputs; **the server learns neither the key material she used nor the mask positions**.
>
> And the crucial property is in the box: the PRF is deterministic, so **two sub-samples are equal after the PRF exactly when they were equal before**. Equality survives; the values themselves are hidden."

**[NEXT SLIDE]**

## Slide 19 — Primitive 2: VOLE (~45 s)

> "Second primitive: **Vector Oblivious Linear Evaluation**.
>
> The problem: this protocol needs an enormous number of correlated random values, and generating them one at a time is far too slow.
>
> VOLE produces the relation on screen — v equals Delta times u plus w. The **sender** knows Delta and v; the **receiver** knows u and w. Neither party alone can reconstruct the relation, yet it holds exactly between them.
>
> That single correlation is the seed for everything downstream."

**[NEXT SLIDE]**

## Slide 20 — Silent VOLE Extension (~40 s)

> "But we need *millions* of these per query, and base VOLEs are expensive.
>
> The trick is **silent extension**: start with a small number of base VOLEs, expand them with a pseudorandom generator, then pass them through a linear code whose security rests on the **LPN assumption** — Learning Parity with Noise.
>
> The result is n correlations from m seeds, where n is vastly larger than m — and they're computationally indistinguishable from genuinely random ones. That's what makes this practical rather than theoretical."

**[NEXT SLIDE]**

## Slide 21 — Primitive 3: Vector Ring-VOLE (~45 s)

> "Third primitive. The matching step needs to multiply **polynomials**, and plain VOLE only gives scalar correlations.
>
> Vector Ring-VOLE lifts it using the evaluation-point trick: a polynomial of degree d is completely determined by its values at 2d+1 points. So instead of multiplying polynomials symbolically, both parties evaluate at a public set of points and apply the VOLE relation **pointwise**.
>
> The receiver ends up with z, the sender with u — the polynomial product, secret-shared between them."

**[NEXT SLIDE]**

## Slide 22 — Ring-VOLE to TLPSI (~55 s)

> "Now the actual matching logic, and this is the elegant part.
>
> **Represent a set by the polynomial whose roots are its elements.** The client builds u-zero with roots at her sub-sampled values. The server builds a Shamir polynomial f, where f evaluated at zero is the label.
>
> They run Ring-VOLE, and the client obtains v-zero equals u-zero times u-one, plus v-one.
>
> Here's the magic identity: **at any point in the intersection, u-zero is zero by construction** — it's a root. So the entire first term vanishes, and the client is left with exactly f at that point: a genuine Shamir share of the label.
>
> At non-intersection points, she gets an unrelated random value.
>
> So every common element hands her one valid share. Collect **t** of them and she interpolates the label. Collect fewer, and the label is **information-theoretically hidden** — not computationally hard, *impossible*."

**[NEXT SLIDE]**

## Slide 23 — The Full Protocol (~45 s)

> "Putting it together: client and server both push their codes through the masked OPRF; TLPSI compares the resulting sets and accepts a record when at least t of the T sub-samples intersect. Out comes either a label, or bottom — nothing.
>
> And the intuition at the bottom is the bridge back to Prabhu's section: **if the Hamming distance is small, enough sub-samples collide by chance to clear the threshold.** Closeness in Hamming space becomes threshold intersection in set space. That's the trick that makes a fuzzy biometric work inside an exact cryptographic primitive."

**[NEXT SLIDE]**

## Slide 24 — Security Analysis (~40 s)

> "Formally: Neural-PSI securely realizes the FLPSI functionality in the hybrid model, against **semi-honest** adversaries — parties who follow the protocol but try to learn from what they see.
>
> The argument has three parts. **One**: the CNN runs strictly locally, so no raw biometric ever leaves the client. **Two**: the server only ever sees PRF-masked sub-samples — never the code itself. **Three**: the FLPSI security argument composes via a standard hybrid argument.
>
> And I'll be upfront: **semi-honest is a real limitation**. Extending to malicious security is on our future-work list.
>
> Prabhu will now show you all of this actually running."

**[HANDOVER → PRABHU] [NEXT SLIDE]**

---

# PART 4 — PRABHU: LIVE DEMO (Slide 25, ~4 min)

> **Setup:** app already running, browser on the **Server database** tab, slide 25 showing.

## 4.1 Framing (~25 s)

> "The problem with fingerprint authentication in the cloud is that the server has to *compare* your fingerprint against a database — but a fingerprint is the one password you can never change. The previous system, Blind-Touch, solved this with homomorphic encryption. It works, but it needs a 117-megabyte key and it stops scaling at around five thousand users.
>
> Our system, Neural-PSI, replaces the encryption with private set intersection. Let me show you the whole pipeline running."

## 4.2 The server database (~30 s)

**[Point at the Server database tab]**

> "This is what the server stores. Forty enrolled fingers. And for each one, the server holds *this* — a 128-bit code. Sixteen bytes.
>
> That's it. No image. No neural network output. No encryption key. Blind-Touch needed about 1.7 kilobytes per user plus that 117-megabyte key; we need sixteen bytes and no key at all."

## 4.3 Genuine authentication (~70 s)

**[Authentication tab → leave on Genuine → Run authentication]**

> "Now a real person walks up, presents a finger, and claims an identity — the system verifies the finger against that claimed record. This is an enrolled finger, but this is a fresh capture — a different scan, with distortion, so it's not pixel-identical to what's stored."

**[Point to Stage 1]**

> "Stage one: the CNN turns each fingerprint image into just sixteen numbers. That runs on the client, on the user's own device — the raw image never leaves."

**[Point to Stage 2]**

> "Stage two is our main contribution. Those sixteen floating-point numbers become a 128-bit code. Each bit answers one question: 'which side of a random dividing plane does this fingerprint fall on?' Similar fingerprints land on the same side of most planes."

**[Point to Stage 3 / the bar chart]**

> "And here's the payoff. Look at the distances. The claimed record — the green bar — is about eight bits different out of 128. Every other person in the database is around sixty-four — half the bits. That's an enormous gap, and that gap is the whole reason this works.
>
> Access granted, label released."

## 4.4 Impostor (~30 s)

**[Switch to Impostor → Run]**

> "Now someone who never enrolled, presenting their finger and claiming to be one of our users. Same pipeline. But look at the distance to the record they're claiming — it's up around eighty bits, nowhere near the threshold. Access denied.
>
> And notice what the server learned in the process: nothing. Not the fingerprint, not the embedding, not the code — not even whether a match occurred."

## 4.5 Real cryptography (~25 s)

**[Sidebar → Stage 3 backend → Real cryptography → Run]**

> "One last thing. Everything so far used our mathematical model of the protocol. Let me switch to the *actual* cryptography — real oblivious PRF, garbled circuits, VOLE, Shamir secret sharing.
>
> Same decision. Around a hundred milliseconds. This is the real protocol, not a simulation."

## 4.6 Close (~20 s)

> "So: sixteen bytes per finger instead of 1.7 kilobytes, no 117-megabyte key, and a linear scan that keeps going where homomorphic encryption gets expensive — while staying within about one and a half percentage points of the accuracy ceiling.
>
> Let me put the actual numbers on screen."

**[NEXT SLIDE]**

---

# PART 5 — PRABHU: RESULTS & CLOSE (Slides 26–29, ~2.5 min)

## Slide 26 — Final Results & Accuracy (~50 s)

> "This is the accuracy ladder, on held-out SOCOFing identities the network never saw in training.
>
> The top row is the **floating-point ceiling — 0.33%**. That's what homomorphic encryption matches on, and nothing downstream of it can do better.
>
> Naive sign: 10%. ITQ: 5.5%. And **Super-Bit at 1.88%** — within about one and a half points of the ceiling, using a 16-byte code.
>
> The last row is the whitening step we mentioned — measured, and rejected.
>
> One caveat I want to state plainly: this is SOCOFing's Altered-Easy protocol, which is deliberately an *easy* benchmark. Cross-sensor testing is future work, and I'd rather say that than have it asked."

**[NEXT SLIDE]**

## Slide 27 — Latency, Storage & Communication (~55 s)

> "And this is the efficiency comparison against Blind-Touch — with one important correction to how these are usually compared.
>
> Blind-Touch's headline number is 650 milliseconds, but that's across **three cluster servers**. On a **single server** — the fair comparison — their own paper reports **1,334 milliseconds**. Ours is **824 on one machine**. So on equal hardware we're roughly **1.6 times faster**.
>
> Storage: **1.67 kilobytes per user versus 16 bytes**, and their **117-megabyte key disappears entirely**.
>
> And we should own the one row that goes the other way: **communication**. We move 7.8 megabytes per query at five thousand users, versus their 856-kilobyte ciphertext. It's linear — about 1.54 kilobytes per record. What gives us confidence that's correct rather than a bug is that it lands almost exactly on the figures published for this protocol: 153 megabytes and 1.5 gigabytes at a hundred thousand and a million users."

**[NEXT SLIDE]**

## Slide 28 — Conclusion & Future Work (~45 s)

> "To conclude. We've shown fingerprint authentication **without revealing the biometric** and **without a trusted third party**. The binarization bridge is what makes a neural network and a set-intersection protocol compatible at all — that's our main technical contribution. And swapping homomorphic encryption for VOLE and Shamir gives 16 bytes per user, no 117-megabyte key, and sub-second latency at five thousand users, with a scan that's linear and shardable beyond that.
>
> Four directions forward. **Larger and harder datasets** — cross-sensor evaluation is the real credibility test. **The algorithm** — our ablation shows accuracy saturates at 128 bits, which means the bottleneck is now the 16-dimensional embedding, not the quantizer; widening it is the highest-value lever. **Parallel processing** — the scan is embarrassingly parallel and we haven't sharded it yet. And **the paper** — completing the formal security proof and an integrated prototype."

**[NEXT SLIDE]**

## Slide 29 — Thank You (~15 s)

> "That's Neural-PSI. Thank you to our mentors and to QNu Labs. We're happy to take questions."

---

# Q&A — Prep Notes

| Question | Who | Answer |
|---|---|---|
| "How accurate really?" | P | 1.88% EER binary; 0.33% float ceiling. Binarization costs ~1.5 points and buys the whole efficiency story. |
| "Is 5,000 a hard cap for Blind-Touch?" | A | **No** — be careful. 512 is per *ciphertext*; the paper explicitly handles N>512 and N>8,192. 5,000 is just their evaluation scale. Their cost grows as ⌈N/512⌉ HE comparisons. |
| "Why 16 dimensions?" | P | Inherited from Blind-Touch — 8,192 CKKS slots ÷ 16 = 512 users/ciphertext. We keep it for compatibility. It's now our accuracy bottleneck. |
| "Why 128 bits?" | P | Measured 64/128/256 → 2.38 / 1.88 / 1.89%. Saturates at 128 because a 16-D source carries ~16 bits of information. Backend also compiles a 128-bit circuit. |
| "Is the binary code reversible?" | R | Yes — and we say so. 128 sign measurements *over*-determine a 16-D vector, so 1-bit compressed sensing recovers its direction. **Privacy comes from the OPRF hiding the code, never from the quantizer.** |
| "What's the leakage exactly?" | R | Server: nothing. Client: the label on a match, plus which sub-samples collided. |
| "Malicious security?" | R | Not yet — semi-honest only. Roughly 2× communication via cut-and-choose. Future work. |
| "Real crypto or simulation?" | P | Both — and the demo shows the real binary. Real FAR 4.43e-2 vs predicted 4.45e-2 over 9,900 impostor decisions. |
| If an impostor is accepted live | P | *"That's the 4.5% false-accept rate at this operating point, exactly as reported. Watch — "* then raise **w** in the sidebar and re-run; it rejects. Owning it reads as rigour. |
| "Why verify against a claimed identity, not search the whole database?" | P | This is 1:1 verification, the operating point the paper validates (w=14, t=2 → FAR 4.5%). 1:N identification searches every record, so a per-comparison error rate compounds with database size and needs a stricter point or multiple fingers — that's noted as future work. |

---

# Timing Card

| | Cumulative |
|---|---|
| Ayan ends (slide 10) | **7:00** |
| Prabhu ends (slide 15) | **11:00** |
| Rineet ends (slide 24) | **17:00** |
| Demo ends (slide 25) | **20:15** |
| Close (slide 29) | **22:45** |

**If running long:** compress slides 19–21 (VOLE primitives) — state the relation and move on. That recovers ~1.5 minutes.

**If running short (most likely):** expand slide 22 (the zero-polynomial trick — the most elegant idea in the talk), and in the demo run a second genuine case with a different distortion type (Obliteration vs Z-cut) to show robustness. Each adds ~40 s.
