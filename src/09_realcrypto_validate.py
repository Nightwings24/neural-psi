"""
Step 9: REAL-CRYPTO validation of the Super-Bit bridge.

Everything in Tier-1/Tier-2 was validated through the *faithful Python port* of flash-psi
(flpsi_match.SubSampler) + the exact closed-form sub-sampling model q(H)=C(d-H,w)/C(d,w).
This script closes the last gap: it pushes our REAL held-out 224-model Super-Bit codes through
the ACTUAL Rust crypto binary (masked-OPRF + garbled-circuit + VOLE + Shamir) and checks that
the real accept/reject decisions match the closed-form prediction ON THE SAME PAIRS. If they
agree, the surrogate we've been trusting is faithful and the end-to-end system is real-crypto-correct.

Protocol per run (the `fingerprint` binary takes ONE query + an m-entry db, returns matched indices):
  db    = enrollment codes (Real print) of a batch of B held-out fingers
  query = the probe code (Altered print) of one finger
  -> a run yields 1 genuine decision (own index) + (B-1) impostor decisions.
Run all B probes as queries => B genuine + B*(B-1) impostor decisions through real crypto.

Op-point: weight=14, T=64, t=2 (the Tier-2 single-finger point). Plain Super-Bit, no whitening.
"""
import argparse
import os
import subprocess
import tempfile
from math import comb

import numpy as np
import torch

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig, hamming

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BIN = os.environ.get("FLASH_PSI_BIN", os.path.abspath(
    os.path.join(HERE, "..", "crypto", "flash-psi", "target", "release", "fingerprint")))
D, T, IMG, CKPT = 128, 64, 224, "feature_model_224.pt"


# ---------- closed-form flash-psi accept probability (same model as 08_tier2_fusion.py) ----------
def q_clean(H, weight, d=D):
    H = int(H)
    if d - H < weight:
        return 0.0
    return comb(d - H, weight) / comb(d, weight)


def binom_tail_ge(n, k, p):
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return float(sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def accept_prob(H, weight, t, T_sub=T):
    return binom_tail_ge(T_sub, t, q_clean(int(H), weight))


# ---------- embedding ----------
def embed_all(model, x, batch=256):
    model.eval(); dev = next(model.parameters()).device; out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def bits_to_str(code):
    return "".join("1" if int(b) else "0" for b in code)


def run_fingerprint(query_code, db_codes, weight, t, ss=T, timeout=120):
    """Call the real Rust FLPSI binary; return (matched_indices set, duration_ms)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        path = f.name
        f.write(f"{len(db_codes)} {D}\n")
        f.write(bits_to_str(query_code) + "\n")
        for row in db_codes:
            f.write(bits_to_str(row) + "\n")
    try:
        out = subprocess.run([BIN, path, "-w", str(weight), "-s", str(ss), "-t", str(t)],
                             capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise RuntimeError(f"fingerprint failed: {out.stderr.strip()[:400]}")
    matched, dur = set(), float("nan")
    for line in out.stdout.splitlines():
        if line.startswith("MATCHED_INDICES"):
            body = line.split("[", 1)[1].rsplit("]", 1)[0].strip()
            matched = set(int(x) for x in body.split(",")) if body else set()
        elif line.startswith("DURATION_MS"):
            dur = float(line.split()[1])
    return matched, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=60, help="db size B (fingers per run)")
    ap.add_argument("--queries", type=int, default=0, help="how many probes to run (0 = all B)")
    ap.add_argument("--weight", type=int, default=14)
    ap.add_argument("--t", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="recompute codes even if cached")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if not os.path.exists(BIN):
        raise SystemExit(f"real FLPSI binary not found: {BIN}\n  build it: see crypto/README.md (set FLASH_PSI_BIN to override)")

    # ---- codes: load from cache, or embed + Super-Bit-encode held-out test fingers ----
    cache = os.path.join(DATA, "sb_codes_224.npz")
    if os.path.exists(cache) and not args.refresh:
        z = np.load(cache, allow_pickle=True)
        fingers, enr, qry = list(z["fingers"]), z["enr"], z["qry"]
        print(f"[rc] loaded {len(fingers)} cached Super-Bit codes <- {cache}")
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = FeatureModel(img=IMG).to(dev)
        model.load_state_dict(torch.load(f"{DATA}/{CKPT}", map_location=dev))
        print(f"[rc] model={CKPT} on {dev}")

        def L(base):
            p = f"{DATA}/{base}_{IMG}.npy"
            return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{base}.npy", allow_pickle=True)
        x_real, ids_real = L("x_real"), L("ids_real")
        x_probe, ids_probe = L("x_probe"), L("ids_probe")
        ids_test = set(L("ids_test").tolist())

        train_real = np.stack([x for i, x in zip(ids_real, x_real) if i not in ids_test])
        quant = NeuralPSIQuantizer.fit(embed_all(model, train_real), np.ones(16),
                                       QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
        real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
        prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
        fingers = sorted(set(real_by) & set(prb_by))
        enr = np.stack([quant.transform(embed_all(model, real_by[i][None])[0]) for i in fingers]).astype(np.uint8)
        qry = np.stack([quant.transform(embed_all(model, prb_by[i][None])[0]) for i in fingers]).astype(np.uint8)
        np.savez(cache, fingers=np.array(fingers, dtype=object), enr=enr, qry=qry)
        print(f"[rc] encoded {len(fingers)} held-out fingers -> cached {cache}")

    # ---- pick a batch of B fingers as the database ----
    B = min(args.batch, len(fingers))
    sel = rng.choice(len(fingers), size=B, replace=False)
    sel.sort()
    db = enr[sel]                                   # enrollment codes (Real)  -> db rows 0..B-1
    db_fingers = [fingers[k] for k in sel]
    nq = B if args.queries <= 0 else min(args.queries, B)
    print(f"[rc] binary={BIN}")
    print(f"[rc] db size B={B} | running nq={nq} probe queries | op-point weight={args.weight} t={args.t} T={T}")
    print(f"[rc] this is the REAL masked-OPRF + GC + VOLE + Shamir protocol (not the surrogate)\n")

    # ---- run real crypto: each probe i (its db index is i, since db == enr[sel]) ----
    gen_real, gen_pred, gen_H = [], [], []
    imp_real, imp_pred, imp_H = [], [], []
    durs = []
    for qi in range(nq):
        query = qry[sel[qi]]                        # probe (Altered) of the same finger as db[qi]
        matched, dur = run_fingerprint(query, db, args.weight, args.t)
        durs.append(dur)
        for j in range(B):
            H = int(hamming(query, db[j]))
            acc = int(j in matched)
            p = accept_prob(H, args.weight, args.t)
            if j == qi:                             # genuine: probe vs its own enrollment
                gen_real.append(acc); gen_pred.append(p); gen_H.append(H)
            else:                                   # impostor: probe vs a different finger's enrollment
                imp_real.append(acc); imp_pred.append(p); imp_H.append(H)
        if (qi + 1) % 10 == 0 or qi == nq - 1:
            print(f"  [{qi+1:>3}/{nq}] done | last run {dur:6.1f} ms | "
                  f"running real TAR={np.mean(gen_real)*100:5.1f}% real FAR={np.mean(imp_real):.2e}")

    gen_real, gen_pred, gen_H = map(np.array, (gen_real, gen_pred, gen_H))
    imp_real, imp_pred, imp_H = map(np.array, (imp_real, imp_pred, imp_H))

    print("\n" + "=" * 74)
    print("  REAL-CRYPTO vs CLOSED-FORM — Super-Bit codes through the actual flash-psi binary")
    print("=" * 74)
    print(f"  decisions: {len(gen_real)} genuine + {len(imp_real)} impostor  (B={B}, {nq} runs)")
    print(f"  per-run latency: mean {np.nanmean(durs):.1f} ms  (m={B}, full setup+online each run)")
    print(f"  genuine  Hamming: mean {gen_H.mean():5.2f} (max {gen_H.max()})   [of {D}]")
    print(f"  impostor Hamming: mean {imp_H.mean():5.2f} (min {imp_H.min()})   [of {D}]")
    print("-" * 74)
    print(f"  {'':<10}{'REAL (crypto)':>18}{'PREDICTED (q-model)':>22}")
    print(f"  {'TAR':<10}{np.mean(gen_real)*100:>16.2f}%{np.mean(gen_pred)*100:>21.2f}%")
    print(f"  {'FAR':<10}{np.mean(imp_real):>17.2e}{np.mean(imp_pred):>22.2e}")
    print("-" * 74)
    # confident-regime correctness: where the model is near-certain, real MUST agree
    conf_acc = gen_pred > 0.99
    conf_rej = imp_pred < 0.01
    if conf_acc.sum():
        print(f"  confident-ACCEPT pairs (pred>0.99): {int((gen_real[conf_acc]==1).sum())}/{int(conf_acc.sum())} really accepted")
    if conf_rej.sum():
        print(f"  confident-REJECT pairs (pred<0.01): {int((imp_real[conf_rej]==0).sum())}/{int(conf_rej.sum())} really rejected")
    # accept-rate vs Hamming: real empirical vs closed-form curve
    print("-" * 74)
    print("  accept-rate by Hamming bucket (real empirical  vs  closed-form q-model):")
    allH = np.concatenate([gen_H, imp_H]); allR = np.concatenate([gen_real, imp_real])
    edges = [0, 10, 20, 25, 30, 35, 40, 50, 129]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (allH >= a) & (allH < b)
        if m.sum() == 0:
            continue
        pred = np.mean([accept_prob(h, args.weight, args.t) for h in allH[m]])
        print(f"     H in [{a:>3},{b:>3}): n={m.sum():>5}  real={allR[m].mean()*100:6.2f}%  pred={pred*100:6.2f}%")
    print("=" * 74)

    # ---- persist a report ----
    rep = os.path.join(HERE, "..", "docs", "results", "real-crypto-validation.md")
    rep = os.path.abspath(rep)
    with open(rep, "w") as f:
        f.write("# Real-crypto validation — Super-Bit codes through the actual flash-psi binary\n\n")
        f.write("Our held-out 224-model Super-Bit codes pushed through the REAL masked-OPRF + garbled-circuit\n")
        f.write("+ VOLE + Shamir protocol (`crypto/fingerprint.rs`), compared against the\n")
        f.write("closed-form sub-sampling model we used in Tier-1/Tier-2. Agreement => the surrogate is faithful\n")
        f.write("and the bridge is real-crypto-correct.\n\n")
        f.write(f"- Operating point: weight={args.weight}, t={args.t}, T={T}; plain Super-Bit (no whitening), d={D}.\n")
        f.write(f"- Database B={B} held-out enrollment (Real) codes; {nq} probe (Altered) queries.\n")
        f.write(f"- Decisions: {len(gen_real)} genuine + {len(imp_real)} impostor, through real crypto.\n")
        f.write(f"- Per-run latency: mean {np.nanmean(durs):.1f} ms (full setup+online, m={B}).\n\n")
        f.write("| metric | REAL (crypto) | PREDICTED (q-model) |\n|---|---|---|\n")
        f.write(f"| TAR | {np.mean(gen_real)*100:.2f}% | {np.mean(gen_pred)*100:.2f}% |\n")
        f.write(f"| FAR | {np.mean(imp_real):.2e} | {np.mean(imp_pred):.2e} |\n")
        f.write(f"| genuine Hamming (mean/max) | {gen_H.mean():.2f} / {gen_H.max()} | — |\n")
        f.write(f"| impostor Hamming (mean/min) | {imp_H.mean():.2f} / {imp_H.min()} | — |\n\n")
        f.write("Accept-rate by Hamming bucket (real vs closed-form):\n\n| H bucket | n | real | predicted |\n|---|---|---|---|\n")
        for a, b in zip(edges[:-1], edges[1:]):
            m = (allH >= a) & (allH < b)
            if m.sum() == 0:
                continue
            pred = np.mean([accept_prob(h, args.weight, args.t) for h in allH[m]])
            f.write(f"| [{a},{b}) | {m.sum()} | {allR[m].mean()*100:.2f}% | {pred*100:.2f}% |\n")
    print(f"[rc] wrote report -> {rep}")


if __name__ == "__main__":
    main()
