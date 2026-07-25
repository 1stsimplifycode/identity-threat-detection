"""Sigma protocols, made non-interactive via Fiat-Shamir.

A Sigma protocol is a 3-move interactive proof (commit, challenge,
response) with three properties: completeness (an honest prover always
convinces an honest verifier), special soundness (a cheating prover who
doesn't know the secret can answer at most one challenge value per
commitment, so cannot pass except with negligible probability), and
honest-verifier zero-knowledge (the transcript can be simulated without
knowing the secret, given the challenge in advance -- meaning it reveals
nothing beyond the truth of the statement).

Fiat-Shamir replaces the verifier's random challenge with a hash of the
prover's first message (and any public context), computed by the prover
itself. This turns the interactive protocol into a single message the
prover can compute once, offline, and anyone can verify later without a
live prover -- which is exactly what letting `dashboard/prepare_data.py`
generate proofs once and `dashboard/app.py` verify them on every page
load requires. The security argument for this transform (in the "random
oracle model", i.e. treating SHA-256 as an idealized random function)
is standard and well-studied; it is not something this module invents.

Two protocols live here:

- `prove_knowledge_of_opening` / `verify_knowledge_of_opening`: a
  2-generator Schnorr proof that the prover knows *some* (value,
  blinding) opening of a Pedersen commitment. General-purpose building
  block.
- `prove_bit` / `verify_bit`: the Cramer-Damgard-Schoenmakers (1994)
  OR-proof that a commitment opens to 0 *or* 1, without revealing which.
  This is the piece `range_proof.py` uses once per bit of the
  (score - threshold) difference.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from zkp.group import G, H, P, Q, mod_exp
from zkp.pedersen import Commitment


def fiat_shamir_challenge(*elements: int, context: bytes) -> int:
    """Hash every public group element in the transcript, plus `context`
    (which binds the proof to whatever it's *about* -- a record ID, a
    model name, a threshold -- so a proof generated for one detection
    can't be replayed as if it were about a different one), down to a
    single challenge in Z_Q.
    """
    hasher = hashlib.sha256()
    hasher.update(context)
    for element in elements:
        hasher.update(element.to_bytes((element.bit_length() + 7) // 8 or 1, "big"))
    return int.from_bytes(hasher.digest(), "big") % Q


@dataclass(frozen=True)
class OpeningProof:
    a: int  # prover's first message, G^kv * H^kr
    sv: int  # response for the value exponent
    sr: int  # response for the blinding exponent


def prove_knowledge_of_opening(
    value: int, blinding: int, commitment: Commitment, context: bytes
) -> OpeningProof:
    from zkp.group import random_exponent

    kv, kr = random_exponent(), random_exponent()
    a = (mod_exp(G, kv) * mod_exp(H, kr)) % P
    e = fiat_shamir_challenge(a, commitment.value, context=context)
    sv = (kv + e * value) % Q
    sr = (kr + e * blinding) % Q
    return OpeningProof(a, sv, sr)


def verify_knowledge_of_opening(
    commitment: Commitment, proof: OpeningProof, context: bytes
) -> bool:
    e = fiat_shamir_challenge(proof.a, commitment.value, context=context)
    lhs = (mod_exp(G, proof.sv) * mod_exp(H, proof.sr)) % P
    rhs = (proof.a * mod_exp(commitment.value, e)) % P
    return lhs == rhs


@dataclass(frozen=True)
class BitProof:
    """CDS OR-proof that `commitment` opens to 0 or to 1.

    Structurally symmetric between the two branches on purpose: a
    verifier (or an attacker inspecting the proof) cannot tell which
    branch was the "real" one and which was simulated -- that symmetry
    *is* the zero-knowledge property for this proof.
    """

    a0: int
    a1: int
    e0: int
    e1: int
    s0: int
    s1: int


def prove_bit(bit: int, blinding: int, commitment: Commitment, context: bytes) -> BitProof:
    if bit not in (0, 1):
        raise ValueError("prove_bit only accepts bit in {0, 1}")
    from zkp.group import mod_inverse, random_exponent

    # Statement 0: commitment == H^r0  (true iff bit == 0, r0 = blinding)
    # Statement 1: commitment * G^-1 == H^r1  (true iff bit == 1, r1 = blinding)
    y0 = commitment.value
    y1 = (commitment.value * mod_inverse(G, P)) % P

    if bit == 0:
        # Real branch 0: honest first move.
        k0 = random_exponent()
        a0 = mod_exp(H, k0)
        # Simulated branch 1: pick the challenge and response first, then
        # solve backwards for the first message that makes them consistent.
        e1 = random_exponent()
        s1 = random_exponent()
        a1 = (mod_exp(H, s1) * mod_exp(y1, Q - e1)) % P
        e = fiat_shamir_challenge(a0, a1, y0, y1, context=context)
        e0 = (e - e1) % Q
        s0 = (k0 + e0 * blinding) % Q
    else:
        k1 = random_exponent()
        a1 = mod_exp(H, k1)
        e0 = random_exponent()
        s0 = random_exponent()
        a0 = (mod_exp(H, s0) * mod_exp(y0, Q - e0)) % P
        e = fiat_shamir_challenge(a0, a1, y0, y1, context=context)
        e1 = (e - e0) % Q
        s1 = (k1 + e1 * blinding) % Q

    return BitProof(a0, a1, e0, e1, s0, s1)


def verify_bit(commitment: Commitment, proof: BitProof, context: bytes) -> bool:
    from zkp.group import mod_inverse

    y0 = commitment.value
    y1 = (commitment.value * mod_inverse(G, P)) % P

    e = fiat_shamir_challenge(proof.a0, proof.a1, y0, y1, context=context)
    if (proof.e0 + proof.e1) % Q != e % Q:
        return False

    lhs0 = mod_exp(H, proof.s0)
    rhs0 = (proof.a0 * mod_exp(y0, proof.e0)) % P
    lhs1 = mod_exp(H, proof.s1)
    rhs1 = (proof.a1 * mod_exp(y1, proof.e1)) % P
    return lhs0 == rhs0 and lhs1 == rhs1
