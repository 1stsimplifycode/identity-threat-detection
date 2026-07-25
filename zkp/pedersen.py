"""Pedersen commitments: `C = G^value * H^blinding mod P`.

Two properties matter here, and why:

- **Perfectly hiding**: for a fixed `C`, every possible `value` has some
  `blinding` that opens `C` to it, uniformly. So `C` alone leaks nothing
  about `value` -- not even to a computationally unbounded adversary.
  This is what lets the dashboard show "score >= threshold: verified"
  without ever exposing the score itself (the analyst-privacy use case).
- **Computationally binding**: the committer cannot later open `C` to a
  *different* value than the one they committed to, unless they can solve
  a discrete-log problem in this group (believed hard). This is what
  makes the proof mean something -- a prover can't commit to one score
  and then claim a different one passed the threshold check.

`commit()` returns both the commitment and the blinding factor (the
caller must keep the blinding factor + value secret; only the commitment
is ever published). Commitments are additively homomorphic in `value`:
`commit(a, r1) * commit(b, r2) == commit(a + b, r1 + r2)` (all mod P for
the product, mod Q for the exponent sum) -- the range proof in
`range_proof.py` leans on this directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from zkp.group import G, H, P, Q, mod_exp, random_exponent


@dataclass(frozen=True)
class Commitment:
    value: int  # C, an element of the group (public)

    def __mul__(self, other: "Commitment") -> "Commitment":
        return Commitment((self.value * other.value) % P)

    def pow(self, exponent: int) -> "Commitment":
        return Commitment(mod_exp(self.value, exponent % Q))

    def inverse(self) -> "Commitment":
        return Commitment(mod_exp(self.value, Q - 1))


def commit(value: int, blinding: int | None = None) -> tuple[Commitment, int]:
    """Commit to `value` (any integer, reduced mod Q). Returns (C, blinding)
    -- caller keeps `blinding` secret. Pass an explicit `blinding` only
    when you need two commitments to share a specific blinding
    relationship (the range proof does this); otherwise let it draw a
    fresh random one.
    """
    if blinding is None:
        blinding = random_exponent()
    c = (mod_exp(G, value % Q) * mod_exp(H, blinding % Q)) % P
    return Commitment(c), blinding


def verify_opening(commitment: Commitment, value: int, blinding: int) -> bool:
    """Recompute the commitment from a claimed (value, blinding) opening
    and check it matches. Only meaningful when the verifier is *given*
    the opening (e.g. in a test, or a non-ZK audit trail) -- the whole
    point of the proofs in this package is to avoid ever needing to
    reveal the opening.
    """
    expected = (mod_exp(G, value % Q) * mod_exp(H, blinding % Q)) % P
    return expected == commitment.value
