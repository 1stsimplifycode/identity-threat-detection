"""Threshold proofs: "this committed value is >= a public threshold", without
revealing the value.

The technique is bit-decomposition + homomorphic recombination, built
entirely out of the two primitives in `sigma.py`:

1. Fix a bit width `n_bits` big enough to hold the largest possible
   difference (`value - threshold`) for this deployment's score scale
   (see `to_fixed_point` below for how a float score becomes an
   integer). This is a public parameter, not secret -- it bounds what
   the proof can express, the same way choosing a field size bounds a
   SNARK circuit.
2. The prover computes `diff = value - threshold`, which must be a
   non-negative integer `< 2**n_bits` for the proof to even be
   constructible -- if `value < threshold`, there IS no valid proof,
   which is exactly the point (the prover cannot fake exceeding a
   threshold they didn't reach).
3. Commit to each bit of `diff` separately, and prove each is 0 or 1
   using `prove_bit` (CDS OR-proof) from `sigma.py`.
4. Because Pedersen commitments are additively homomorphic
   (`commit(a) * commit(b) == commit(a+b)`, see `pedersen.py`), the
   product of the bit commitments raised to their place values
   (`prod(C_i ** 2**i)`) is itself a commitment to `diff` -- with a
   *derived* blinding factor the verifier can compute as the same
   product-of-powers over the blinding factors. So step 5 below just
   checks that this recombined commitment equals `commitment_to_value *
   commitment_to_threshold^-1`, tying the per-bit proofs back to the
   original value commitment without ever opening it.
5. The verifier: checks every per-bit OR-proof, then checks the
   homomorphic recombination identity. If both hold, `diff` is
   guaranteed to be a non-negative `n_bits`-bit number, which means
   `value >= threshold` -- QED, and the verifier never learned `value`,
   `diff`, or any individual bit.

This is a standard (if simplified/unoptimized -- no aggregation tricks
like Bulletproofs' logarithmic-size proofs) range-proof construction.
Proof size and verification cost are O(n_bits); at the `n_bits=32` used
by this project's default config that's 32 CDS proofs, still fast
enough (see the benchmark in the test suite) for this project's scale.
"""
from __future__ import annotations

from dataclasses import dataclass

from zkp.group import P, Q, mod_exp
from zkp.pedersen import Commitment, commit
from zkp.sigma import BitProof, prove_bit, verify_bit


def to_fixed_point(score: float, scale: int = 10_000) -> int:
    """Convert a float anomaly score to a bounded non-negative integer
    suitable for bit-decomposition, by multiplying up and truncating.
    `scale=10_000` keeps 4 decimal digits of precision, matching the
    precision already reported in this project's evaluation tables
    (e.g. `precision: 0.5558`) -- no information the dashboard already
    surfaces elsewhere is lost by this rounding.
    """
    if score < 0:
        raise ValueError("scores in this project's [0, 1]-normalized range are expected to be non-negative")
    return int(round(score * scale))


@dataclass(frozen=True)
class ThresholdProof:
    diff_bit_commitments: tuple[int, ...]  # C_i for each bit of (value - threshold)
    diff_bit_proofs: tuple[BitProof, ...]
    n_bits: int


def prove_at_least_threshold(
    value: int, blinding: int, threshold: int, n_bits: int, context: bytes
) -> ThresholdProof:
    """Prove that the committed `value` (whose opening the prover holds:
    `value`, `blinding`) is >= `threshold`, without revealing `value`.

    Raises ValueError if the claim is false or out of the provable range
    -- there is no way to produce a `ThresholdProof` for a false
    statement, which is the whole point.
    """
    diff = value - threshold
    if diff < 0:
        raise ValueError("value is below threshold -- no valid proof exists")
    if diff >= (1 << n_bits):
        raise ValueError(f"value - threshold does not fit in {n_bits} bits")

    bit_commitments: list[int] = []
    bit_blindings: list[int] = []
    bit_proofs: list[BitProof] = []
    for i in range(n_bits):
        bit = (diff >> i) & 1
        c_i, r_i = commit(bit)
        proof_i = prove_bit(bit, r_i, c_i, context=context + i.to_bytes(4, "big"))
        bit_commitments.append(c_i.value)
        bit_blindings.append(r_i)
        bit_proofs.append(proof_i)

    # The blinding factor implied by the homomorphic recombination of the
    # bit commitments must equal (blinding - threshold's-implicit-zero-
    # blinding); since we commit to the threshold with blinding 0 (it's
    # public, no hiding needed), the recombined diff commitment's implied
    # blinding is sum(r_i * 2**i). For the proof to tie back to the
    # original `commitment` in verification, that sum must equal
    # `blinding` itself -- so we don't get to choose the per-bit
    # blindings freely; the last one is solved for.
    implied = sum((bit_blindings[i] * pow(2, i, Q)) % Q for i in range(n_bits)) % Q
    # Adjust the last bit's blinding so the sum matches `blinding` exactly.
    # Recompute that bit's commitment and proof with the corrected blinding.
    last = n_bits - 1
    correction = (blinding - (implied - (bit_blindings[last] * pow(2, last, Q)) % Q)) % Q
    correction = (correction * pow(pow(2, last, Q), -1, Q)) % Q
    bit_blindings[last] = correction
    last_bit = (diff >> last) & 1
    c_last, _ = commit(last_bit, blinding=correction)
    bit_commitments[last] = c_last.value
    bit_proofs[last] = prove_bit(
        last_bit, correction, c_last, context=context + last.to_bytes(4, "big")
    )

    return ThresholdProof(tuple(bit_commitments), tuple(bit_proofs), n_bits)


def verify_at_least_threshold(
    commitment: Commitment, threshold: int, proof: ThresholdProof, context: bytes
) -> bool:
    if len(proof.diff_bit_commitments) != proof.n_bits or len(proof.diff_bit_proofs) != proof.n_bits:
        return False

    for i in range(proof.n_bits):
        c_i = Commitment(proof.diff_bit_commitments[i])
        if not verify_bit(c_i, proof.diff_bit_proofs[i], context=context + i.to_bytes(4, "big")):
            return False

    recombined = 1
    for i in range(proof.n_bits):
        recombined = (recombined * mod_exp(proof.diff_bit_commitments[i], pow(2, i, Q))) % P

    # commitment_to_threshold, with blinding 0 since the threshold is public
    threshold_commitment, _ = commit(threshold, blinding=0)
    expected = (recombined * threshold_commitment.value) % P
    return expected == commitment.value
