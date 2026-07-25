"""The discrete-log group everything else in this package works in.

Modulus: RFC 3526 "MODP Group 14" -- the standardized 2048-bit safe prime
used for decades in IKE/SSH/TLS Diffie-Hellman. We use a well-known,
publicly-audited constant here deliberately, instead of generating our own
prime at runtime: safe primes are sparse (generating one, then checking
(p-1)/2 is also prime, is slow -- verified directly during development,
1024-2048 bit safe-prime search took multiple minutes on this machine), and
a freshly-generated prime nobody else has reviewed is a strictly worse
starting point than a parameter set the wider cryptographic community has
already scrutinized. The constant below was transcribed once and verified
in this session (`p` prime, `(p-1)//2` prime, `p.bit_length() == 2048`)
before being hardcoded -- see the assertions at import time, which repeat
that check every time this module loads, so a future transcription error
anywhere would fail loudly rather than silently producing an insecure
group.

`p - 1 = 2q` with `q` prime (a "safe prime"), so the group of quadratic
residues mod `p` is a subgroup of prime order `q`. Both Pedersen
commitments and the Sigma protocols below need a *prime*-order group (so
that every nonzero element has a well-defined inverse exponent, and so
there's no small-subgroup structure an adversary could exploit) -- that's
why we work in this QR subgroup, not the full multiplicative group mod p.

Two independent generators `G` and `H` of that subgroup are required for
Pedersen commitments (`C = G^v * H^r mod p`): "independent" means nobody
-- including us -- knows `x` such that `H = G^x`, because knowing that
`x` would let you open any commitment to any value you like (break
binding). `G` is derived directly (`2^2 mod p` is standard for this
group). `H` is derived by hashing a fixed, public seed string
("nothing-up-my-sleeve": anyone can recompute it and confirm we didn't
choose `H` to know its relationship to `G`), which is the standard way to
manufacture a second generator nobody has a discrete-log shortcut for.

Caveat, stated plainly: this is a hand-rolled implementation of a
textbook (Pedersen/Chaum-Pedersen/CDS) construction, not a
security-audited library. It is appropriate for a documented prototype
that demonstrates the actual cryptographic technique end-to-end; it is
not a substitute for an audited library (e.g. a maintained Bulletproofs
or Groth16 implementation) before any real deployment. See
docs/zero_knowledge_proofs.md.
"""
from __future__ import annotations

import hashlib

# RFC 3526, "MODP Group 14" (2048-bit). Transcribed and verified prime
# (with (p-1)//2 also prime) during development -- see module docstring.
P = int(
    "FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1"
    "29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD"
    "EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245"
    "E485B576 625E7EC6 F44C42E9 A637ED6B 0BFF5CB6 F406B7ED"
    "EE386BFB 5A899FA5 AE9F2411 7C4B1FE6 49286651 ECE45B3D"
    "C2007CB8 A163BF05 98DA4836 1C55D39A 69163FA8 FD24CF5F"
    "83655D23 DCA3AD96 1C62F356 208552BB 9ED52907 7096966D"
    "670C354E 4ABC9804 F1746C08 CA18217C 32905E46 2E36CE3B"
    "E39E772C 180E8603 9B2783A2 EC07A28F B5C55DF0 6F4C52C9"
    "DE2BCBF6 95581718 3995497C EA956AE5 15D22618 98FA0510"
    "15728E5A 8AACAA68 FFFFFFFF FFFFFFFF".replace(" ", ""),
    16,
)
Q = (P - 1) // 2  # order of the quadratic-residue subgroup we work in

assert P.bit_length() == 2048
assert P == 2 * Q + 1


def _is_prime(n: int) -> bool:
    """Miller-Rabin, deterministic-enough for our fixed constants.

    Only ever called on the two fixed constants above at import time (not
    on caller-supplied data), so a probabilistic test with a healthy
    round count is appropriate -- this isn't gating untrusted input.
    """
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if n % small == 0:
            return n == small
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


assert _is_prime(P), "RFC 3526 modulus failed primality check -- transcription error"
assert _is_prime(Q), "safe-prime cofactor failed primality check -- transcription error"


def _hash_to_subgroup(seed: bytes) -> int:
    """Nothing-up-my-sleeve derivation of a QR-subgroup element from `seed`.

    Concatenates SHA-256(seed || counter) for increasing counter until we
    have >= bit_length(P) bits, reduces mod P, then squares -- squaring
    any element of Z_p^* lands it in the order-`Q` subgroup (since
    `P - 1 = 2Q`), and the odds of that square landing on the identity
    are astronomically small for a random input, so we don't bother
    special-casing it (an accidental identity output here would be
    immediately obvious: every subsequent proof would degenerate).
    """
    digest_bits = hashlib.sha256().digest_size * 8
    needed = P.bit_length()
    chunks: list[bytes] = []
    counter = 0
    bits_so_far = 0
    while bits_so_far < needed:
        chunks.append(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        bits_so_far += digest_bits
        counter += 1
    raw = int.from_bytes(b"".join(chunks), "big") % P
    return pow(raw, 2, P)


G = pow(2, 2, P)
H = _hash_to_subgroup(b"identity-threat-detection/zkp/pedersen-generator-h/v1")

assert G != 1 and H != 1 and G != H
assert pow(G, Q, P) == 1 and pow(H, Q, P) == 1, "generators must have order Q"


def mod_exp(base: int, exponent: int, modulus: int = P) -> int:
    return pow(base % modulus, exponent, modulus)


def mod_inverse(value: int, modulus: int = P) -> int:
    return pow(value, -1, modulus)


def random_exponent() -> int:
    """A uniformly random element of Z_Q -- used for commitment blinding
    factors and Sigma-protocol nonces. Every value that must stay secret
    (blinding factors, prover nonces) is drawn from here, never from a
    non-cryptographic RNG.
    """
    import secrets

    return secrets.randbelow(Q)
