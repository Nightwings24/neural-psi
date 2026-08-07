"""
Faithful Python port of flash-psi's fuzzy sub-sampling matcher (Stage 3).

Mirrors flash-psi/src/subsample.rs:
  - A Mask is a length-d binary vector with EXACTLY `weight` ones (rest zeros),
    positions chosen by a random shuffle.
  - Mask.apply(x) = x & mask  (keep only the `weight` selected bit positions).
  - Two items "sub-match" on a mask when their masked vectors are equal, i.e.
    they agree on every selected position.
  - Items fuzzy-MATCH when the number of sub-matches over T masks is >= t.

This is the plaintext logic the real protocol (masked OPRF + garbled circuits +
Vector Ring-OLE + Shamir) computes securely. Same accept/reject decision, no HE.
"""
import math

import numpy as np


class SubSampler:
    """T masks of exactly `weight` ones over d positions (matches Rust SubSampler)."""

    def __init__(self, d, weight, subsample_count, seed=0):
        assert 0 < weight < d, "need 0 < weight < d"
        max_unique = math.comb(d, weight)
        if subsample_count > max_unique:
            raise ValueError(
                f"can't draw {subsample_count} unique weight-{weight} masks over "
                f"d={d}: only {max_unique} exist")
        rng = np.random.default_rng(seed)
        base = np.array([1] * weight + [0] * (d - weight), dtype=np.int8)
        masks, seen = [], set()
        # draw unique masks (Rust panics on duplicates; we just resample)
        while len(masks) < subsample_count:
            m = base.copy()
            rng.shuffle(m)
            key = m.tobytes()
            if key in seen:
                continue
            seen.add(key)
            masks.append(m)
        self.d = d
        self.weight = weight
        self.masks = np.stack(masks)                 # (T, d)

    def sub_matches(self, a, b):
        """# of masks on which a and b agree across all selected positions."""
        # agree[j] over selected positions of mask m  <=>  (a==b) covers mask m
        eq = (a == b).astype(np.int8)                # (d,)
        # a sub-match on mask m: all selected positions equal
        # == (eq AND mask).sum == weight  <=>  no selected position differs
        return int(np.sum((self.masks * (1 - eq)).sum(1) == 0))

    def is_match(self, a, b, t):
        return self.sub_matches(a, b) >= t
