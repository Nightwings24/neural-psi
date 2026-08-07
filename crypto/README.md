# Cryptographic backend — flash-psi additions

The Fuzzy-Labelled PSI protocol we use is **not our code**. It is the reference implementation
accompanying:

> Dung Bui and Kelong Cong. *Efficient Fuzzy Labeled PSI from Vector Ring-OLE.*
> CANS 2025. ePrint [2025/1470](https://eprint.iacr.org/2025/1470).
> Source: <https://github.com/kc1212/flash-psi>

Rather than vendor someone else's repository, this directory contains **only our additions**:

| File | What it is |
|---|---|
| `fingerprint.rs` | New binary. Runs the real FLPSI protocol on real fingerprint codes read from a text file, and prints the matched record indices and wall-clock duration. |
| `neuralpsi-flash-psi.patch` | Our changes to two upstream files (46 lines): adds `flpsi::setup_flpsi_with_inputs` so the protocol can run on caller-supplied database codes instead of randomly generated ones, and adds offline-phase timing to the `simulation` binary. |

## Build

```bash
git clone https://github.com/kc1212/flash-psi.git
cd flash-psi
git apply ../neuralpsi-flash-psi.patch
cp ../fingerprint.rs src/bin/
cargo build --release --bin fingerprint --bin simulation
```

This produces `target/release/fingerprint`. The Python tooling looks for it at
`crypto/flash-psi/target/release/fingerprint`; set `FLASH_PSI_BIN` to point elsewhere.

## Input format for `fingerprint`

All vectors are 128 characters of `0`/`1`:

```
<m> 128          # m = database size
<query bits>
<db row 1 bits>
...
<db row m bits>
```

Run as `fingerprint <input_file> [-w WEIGHT] [-s SUBSAMPLES] [-t THRESHOLD]`, defaulting to
our validated operating point `w=14, s=64, t=2`.

> Note: the PRF uniqueness check panics on degenerate all-identical codes. Real Super-Bit
> codes are always fine.

## Reproducing the measurements

```bash
# real-crypto correctness (writes docs/results/real-crypto-validation.md)
python src/09_realcrypto_validate.py --batch 100 --queries 100

# communication + latency vs database size
for M in 100 1000 5000 10000 50000; do
  ./target/release/simulation --sim-type flpsi -m $M -w 14 -s 64 -t 2 --track-io --csv
done
```

Upstream flash-psi is distributed under its own licence; see the LICENSE file in that
repository. The two files here are ours and fall under this repository's MIT licence.
