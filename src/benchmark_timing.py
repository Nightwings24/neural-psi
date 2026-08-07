"""
Reproducible timing benchmark: Neural-PSI (flash-psi) vs Blind-Touch (HE).

Runs the REAL flash-psi FLPSI protocol at increasing database sizes m, prints
the scaling curve (it is linear in m), and places it next to Blind-Touch's
published homomorphic-encryption number to make the HE-vs-PSI argument.

Needs only the compiled simulation binary:  flash-psi/target/release/simulation
Run:  python3 benchmark_timing.py            (default up to 100k, ~30-60 s)
      python3 benchmark_timing.py --max 1000000   (warning: needs lots of RAM)

NOTE: flash-psi `t` MUST stay small (paper value 2): its Shamir step is
combinations(t+1), so cost explodes with t. Tune `weight`, never `t`.
"""
import argparse
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "flash-psi", "target", "release", "simulation")

# Blind-Touch HE reference (from the paper/README): ~650 ms for 5,000 records on
# their 4-server NAVER-cloud cluster; CKKS slot-packing caps it near ~5,000.
HE_REF_M = 5000
HE_REF_S = 0.65
HE_CAP = 5000


def run_flpsi(m, weight, ss, t):
    out = subprocess.run(
        [SIM, "--sim-type", "flpsi", "-m", str(m),
         "-w", str(weight), "-s", str(ss), "-t", str(t)],
        capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "8"},
    )
    mobj = re.search(r"total duration:\s*([0-9.]+)(ms|s)", out.stdout)
    if not mobj:
        return None
    val = float(mobj.group(1))
    return val / 1000.0 if mobj.group(2) == "ms" else val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=int, default=14)
    ap.add_argument("--ss", type=int, default=64)
    ap.add_argument("--t", type=int, default=2)
    ap.add_argument("--max", type=int, default=100000)
    args = ap.parse_args()

    if not os.path.exists(SIM):
        raise SystemExit(f"simulation binary not found at {SIM}\n"
                         f"build it: cd flash-psi && cargo build --release")

    sizes = [s for s in [50, 1000, 5000, 10000, 50000, 100000, 1000000] if s <= args.max]
    print(f"flash-psi FLPSI timing (weight={args.weight}, T={args.ss}, t={args.t})\n")
    print(f"  {'DB size m':>12} {'flash-psi':>12} {'per-record':>12}")
    print("  " + "-" * 38)
    per_record = None
    for m in sizes:
        secs = run_flpsi(m, args.weight, args.ss, args.t)
        if secs is None:
            print(f"  {m:>12,} {'(failed)':>12}")
            continue
        per_record = secs / m * 1000  # ms/record
        shown = f"{secs*1000:.0f} ms" if secs < 1 else f"{secs:.2f} s"
        print(f"  {m:>12,} {shown:>12} {per_record:>9.3f} ms")

    # extrapolate to 1M if we didn't measure it
    if per_record is not None and 1000000 not in sizes:
        est = per_record * 1000000 / 1000.0
        print(f"  {'1,000,000':>12} {f'~{est:.0f} s (est)':>12} {per_record:>9.3f} ms")

    print("\n  HE vs PSI (matching at scale):")
    print(f"  {'DB size':>12} {'Blind-Touch (HE)':>22} {'Neural-PSI (flash-psi)':>24}")
    print("  " + "-" * 60)
    rows = [(5000, f"~{HE_REF_S*1000:.0f} ms (4-srv cluster)"),
            (10000, "cant (CKKS slot cap)"),
            (100000, "cant (CKKS slot cap)"),
            (1000000, "cant (CKKS slot cap)")]
    for m, he in rows:
        psi = (f"{per_record*m/1000:.1f} s" if per_record else "-")
        print(f"  {m:>12,} {he:>22} {psi:>24}")
    print(f"\n  => HE caps near {HE_CAP:,}; flash-psi scales linearly "
          f"({per_record:.2f} ms/record)." if per_record else "")


if __name__ == "__main__":
    main()
