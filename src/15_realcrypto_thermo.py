"""
Step 15: REAL-CRYPTO validation of the metric-aware L2-thermometer bridge,
head-to-head with Super-Bit, through the ACTUAL flash-psi binary.

Mirrors 09_realcrypto_validate.py but (a) also runs the thermometer bridge and
(b) builds codes from cached embeddings (dim_ablation_emb_224.npz: gal/prb/train)
so it needs no GPU (the dimension sweep owns the GPU). Same op-point w=14,t=2,T=64.

Goal: confirm the real accept/reject decisions match the closed-form q-model for
BOTH bridges on the same held-out fingers, and report each bridge's real FAR/TAR.
"""
import argparse, os, subprocess, tempfile
from math import comb
import numpy as np
from quantizer import NeuralPSIQuantizer, QuantizerConfig, hamming

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BIN = os.environ.get("FLASH_PSI_BIN", os.path.abspath(
    os.path.join(HERE, "..", "crypto", "flash-psi", "target", "release", "fingerprint")))
D, T = 128, 64


def q_clean(H, w, d=D):
    H = int(H)
    return 0.0 if d - H < w else comb(d - H, w) / comb(d, w)


def binom_tail_ge(n, k, p):
    if k <= 0: return 1.0
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    return float(sum(comb(n, j) * p**j * (1-p)**(n-j) for j in range(k, n+1)))


def accept_prob(H, w, t, T_sub=T):
    return binom_tail_ge(T_sub, t, q_clean(int(H), w))


def bits_to_str(code):
    return "".join("1" if int(b) else "0" for b in code)


def run_fingerprint(query, db, w, t, ss=T, timeout=180):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        path = f.name
        f.write(f"{len(db)} {D}\n"); f.write(bits_to_str(query) + "\n")
        for row in db: f.write(bits_to_str(row) + "\n")
    try:
        out = subprocess.run([BIN, path, "-w", str(w), "-s", str(ss), "-t", str(t)],
                             capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise RuntimeError(f"fingerprint failed: {out.stderr.strip()[:300]}")
    matched, dur = set(), float("nan")
    for line in out.stdout.splitlines():
        if line.startswith("MATCHED_INDICES"):
            body = line.split("[", 1)[1].rsplit("]", 1)[0].strip()
            matched = set(int(x) for x in body.split(",")) if body else set()
        elif line.startswith("DURATION_MS"):
            dur = float(line.split()[1])
    return matched, dur


def superbit_codes(tr, g, p):
    q = NeuralPSIQuantizer.fit(tr, np.ones(16),
                               QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    return q.transform(g).astype(np.uint8), q.transform(p).astype(np.uint8)


def thermo_codes(tr, g, p, K=16, Bp=8, seed=1):
    r = np.random.default_rng(seed); A = r.standard_normal((K, tr.shape[1])); mu = tr.mean(0)
    pj = (tr - mu) @ A.T
    qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)
    f = lambda X: (((X - mu) @ A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(g), f(p)


def validate(name, enr, qry, B, nq, w, t, rng):
    sel = np.sort(rng.choice(len(enr), size=B, replace=False))
    db = enr[sel]
    gen_real, gen_pred, imp_real, imp_pred, gen_H, imp_H, durs = [], [], [], [], [], [], []
    for qi in range(nq):
        query = qry[sel[qi]]
        matched, dur = run_fingerprint(query, db, w, t); durs.append(dur)
        for j in range(B):
            H = int(hamming(query, db[j])); acc = int(j in matched); pr = accept_prob(H, w, t)
            if j == qi: gen_real.append(acc); gen_pred.append(pr); gen_H.append(H)
            else: imp_real.append(acc); imp_pred.append(pr); imp_H.append(H)
        if (qi+1) % 20 == 0 or qi == nq-1:
            print(f"  [{name}] {qi+1}/{nq} | last {dur:.0f}ms | real TAR {np.mean(gen_real)*100:.1f}% FAR {np.mean(imp_real):.2e}")
    gr, gp, ir, ip = map(np.array, (gen_real, gen_pred, imp_real, imp_pred))
    gH, iH = np.array(gen_H), np.array(imp_H)
    return dict(name=name, tar_real=gr.mean(), tar_pred=gp.mean(), far_real=ir.mean(), far_pred=ip.mean(),
                genH=gH.mean(), impH=iH.mean(), n_gen=len(gr), n_imp=len(ir), dur=np.nanmean(durs),
                conf_rej_ok=int((ir[ip < 0.01] == 0).sum()), conf_rej_n=int((ip < 0.01).sum()),
                conf_acc_ok=int((gr[gp > 0.99] == 1).sum()), conf_acc_n=int((gp > 0.99).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--weight", type=int, default=14)
    ap.add_argument("--t", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(BIN):
        raise SystemExit(f"binary not found: {BIN}")
    z = np.load(os.path.join(DATA, "dim_ablation_emb_224.npz"), allow_pickle=True)
    gal, prb, tr = z["gal"].astype(np.float64), z["prb"].astype(np.float64), z["train"].astype(np.float64)
    B = min(args.batch, len(gal)); nq = min(args.queries, B)
    print(f"[rc] binary={BIN}\n[rc] B={B} nq={nq} op-point w={args.weight} t={args.t} T={T} (real masked-OPRF+GC+VOLE+Shamir)\n")

    results = []
    for name, (enr, qry) in [("Super-Bit", superbit_codes(tr, gal, prb)),
                             ("L2-thermo", thermo_codes(tr, gal, prb))]:
        rng = np.random.default_rng(args.seed)   # same finger subset for both bridges
        print(f"--- {name} ---")
        results.append(validate(name, enr, qry, B, nq, args.weight, args.t, rng))

    print("\n" + "=" * 78)
    print("  REAL-CRYPTO head-to-head — same held-out fingers, same op-point, actual binary")
    print("=" * 78)
    print(f"  {'bridge':<12}{'real TAR':>10}{'real FAR':>12}{'pred FAR':>12}{'genH':>7}{'impH':>7}{'ms/run':>9}")
    for r in results:
        print(f"  {r['name']:<12}{r['tar_real']*100:>9.1f}%{r['far_real']:>12.2e}{r['far_pred']:>12.2e}"
              f"{r['genH']:>7.1f}{r['impH']:>7.1f}{r['dur']:>9.1f}")
    print("-" * 78)
    for r in results:
        print(f"  [{r['name']}] confident-REJECT {r['conf_rej_ok']}/{r['conf_rej_n']} really rejected; "
              f"confident-ACCEPT {r['conf_acc_ok']}/{r['conf_acc_n']} really accepted "
              f"({r['n_gen']} gen + {r['n_imp']} imp decisions)")
    print("=" * 78)

    rep = os.path.abspath(os.path.join(HERE, "..", "docs", "results", "real-crypto-thermo.md"))
    with open(rep, "w") as f:
        f.write("# Real-crypto head-to-head — Super-Bit vs L2-thermometer bridge\n\n")
        f.write(f"Both bridges' codes for the SAME {B} held-out fingers pushed through the actual flash-psi\n")
        f.write(f"binary (masked-OPRF+GC+VOLE+Shamir). Op-point w={args.weight}, t={args.t}, T={T}, d={D}.\n")
        f.write(f"{results[0]['n_gen']} genuine + {results[0]['n_imp']} impostor decisions per bridge.\n\n")
        f.write("| bridge | real TAR | real FAR | pred FAR | gen H | imp H | ms/run |\n|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['tar_real']*100:.1f}% | {r['far_real']:.2e} | {r['far_pred']:.2e} "
                    f"| {r['genH']:.1f} | {r['impH']:.1f} | {r['dur']:.1f} |\n")
        f.write("\nConfident-regime correctness (real must agree with the near-certain model):\n\n")
        for r in results:
            f.write(f"- **{r['name']}**: confident-REJECT {r['conf_rej_ok']}/{r['conf_rej_n']}, "
                    f"confident-ACCEPT {r['conf_acc_ok']}/{r['conf_acc_n']}.\n")
    print(f"[rc] wrote {rep}")


if __name__ == "__main__":
    main()
