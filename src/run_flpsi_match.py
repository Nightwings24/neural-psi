"""
Pure-Python flash-psi match verification (no Rust toolchain needed).

Uses YOUR own faithful port `flpsi_match.SubSampler` (week1/flpsi_match.py), which mirrors
flash-psi/src/subsample.rs, to run the Stage-3 fuzzy sub-sampling decision on the exported
code files. Same accept/reject logic as the real protocol (masked OPRF + GC + VOLE + Shamir),
no cryptography - so it confirms the MATCH CORRECTNESS without building the Rust binary.

Place this file in week1/ (next to flpsi_match.py) and run:  python3 run_flpsi_match.py

It checks both bridges, reports per-seed robustness, and compares against ground truth:
  * Super-Bit 128-bit codes  (data/psi_sb/{genuine,impostor}.txt)   params w=24, t=2, T=64
  * ITQ / sign 128-bit codes (data/psi/{itq,sign}_{genuine,impostor}.txt) params w=14, t=3, T=64
"""
import os
import numpy as np

from flpsi_match import SubSampler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_case(path):
    lines = open(path).read().split("\n")
    m, d = map(int, lines[0].split())
    query = np.array([int(c) for c in lines[1]], dtype=np.int8)
    db = [np.array([int(c) for c in lines[2 + i]], dtype=np.int8) for i in range(m)]
    return d, query, db


def ground_truth(psi_dir):
    return int(open(os.path.join(psi_dir, "ground_truth.txt")).read().split()[1])


def run(psi_dir, names, weight, thresh, T=64, seeds=range(10)):
    gt = ground_truth(psi_dir)
    print(f"\n### {os.path.basename(psi_dir)}/  (w={weight}, t={thresh}, T={T}) | ground-truth genuine row = {gt}")
    for name in names:
        path = os.path.join(psi_dir, f"{name}.txt")
        if not os.path.exists(path):
            continue
        d, q, db = load_case(path)
        matchsets = []
        for seed in seeds:
            s = SubSampler(d=d, weight=weight, subsample_count=T, seed=seed)
            matchsets.append([i for i, r in enumerate(db) if s.sub_matches(q, r) >= thresh])
        hds = [int(np.sum(q != r)) for r in db]
        gt_in_all = all(gt in ms for ms in matchsets) if "genuine" in name else None
        clean_all = all(ms == [] for ms in matchsets) if "impostor" in name else None
        print(f"  {name:<16} min-HD={min(hds):>3}/{d} (row {int(np.argmin(hds))}) | "
              f"row{gt} HD={hds[gt]:>3} | match-set sizes={[len(m) for m in matchsets]}")
        if gt_in_all is not None:
            print(f"      -> genuine row {gt} matched in {sum(gt in m for m in matchsets)}/{len(matchsets)} seeds  "
                  f"({'ALWAYS' if gt_in_all else 'NOT always'})")
        if clean_all is not None:
            print(f"      -> impostor rejected in {sum(m==[] for m in matchsets)}/{len(matchsets)} seeds  "
                  f"({'ALWAYS clean' if clean_all else 'some FALSE ACCEPTS'})")


if __name__ == "__main__":
    run(os.path.join(DATA, "psi_sb"), ["genuine", "impostor"], weight=24, thresh=2)
    run(os.path.join(DATA, "psi"), ["itq_genuine", "itq_impostor", "sign_genuine", "sign_impostor"],
        weight=14, thresh=3)
