"""Application layer: binds a threshold proof to a specific detection record.

Everything below `range_proof.py` is generic cryptography. This module is
where that cryptography meets this project's actual data: a flagged
event has a `record_id`, was scored by a named model, and was compared
against that model's configured decision threshold. A `DetectionProof`
packages the commitment and threshold proof together with those three
public identifiers, and -- critically -- folds them into the Fiat-Shamir
context so a proof generated for one (record, model, threshold) cannot
be replayed as if it were about a different one (e.g. pasting record
A's genuine proof onto record B's page).

Two things this proves, matching the two purposes the user asked for:

- **Detection integrity**: the value inside `value_commitment` really
  does exceed `threshold_fixed_point`, and that fact is bound to this
  exact `record_id` + `model_name` + `threshold`. An analyst (or an
  auditor) can verify this without trusting the dashboard's rendering
  code -- the proof stands on its own math.
- **Analyst privacy**: verification never requires opening the
  commitment. The dashboard can show "verified: score exceeds
  threshold" and never display -- and never even hold in memory after
  proof generation -- the raw score, in views where that's desirable.

What this deliberately does NOT prove (see docs/zero_knowledge_proofs.md
for the full discussion): that the committed score was itself correctly
computed by the model from the record's raw features. Proving that would
require a zkML circuit over the actual model (e.g. EZKL/Halo2 for the
XGBoost/tree models), which is out of scope here. This proves integrity
of the *threshold decision given the score*, not correctness of the
*scoring function*.
"""
from __future__ import annotations

from dataclasses import dataclass

from zkp.pedersen import Commitment, commit
from zkp.range_proof import (
    ThresholdProof,
    prove_at_least_threshold,
    to_fixed_point,
    verify_at_least_threshold,
)
from zkp.sigma import BitProof

_DOMAIN = b"identity-threat-detection/zkp/detection-proof/v1"


def _context(record_id: str, model_name: str, threshold_fixed_point: int, scale: int, n_bits: int) -> bytes:
    parts = [
        _DOMAIN,
        record_id.encode("utf-8"),
        model_name.encode("utf-8"),
        str(threshold_fixed_point).encode("ascii"),
        str(scale).encode("ascii"),
        str(n_bits).encode("ascii"),
    ]
    return b"|".join(parts)


@dataclass(frozen=True)
class DetectionProof:
    record_id: str
    model_name: str
    threshold_fixed_point: int
    scale: int
    n_bits: int
    value_commitment: int
    threshold_proof: ThresholdProof

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "model_name": self.model_name,
            "threshold_fixed_point": self.threshold_fixed_point,
            "scale": self.scale,
            "n_bits": self.n_bits,
            "value_commitment": self.value_commitment,
            "threshold_proof": {
                "diff_bit_commitments": list(self.threshold_proof.diff_bit_commitments),
                "diff_bit_proofs": [
                    {"a0": p.a0, "a1": p.a1, "e0": p.e0, "e1": p.e1, "s0": p.s0, "s1": p.s1}
                    for p in self.threshold_proof.diff_bit_proofs
                ],
                "n_bits": self.threshold_proof.n_bits,
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "DetectionProof":
        tp = data["threshold_proof"]
        return DetectionProof(
            record_id=data["record_id"],
            model_name=data["model_name"],
            threshold_fixed_point=data["threshold_fixed_point"],
            scale=data["scale"],
            n_bits=data["n_bits"],
            value_commitment=data["value_commitment"],
            threshold_proof=ThresholdProof(
                diff_bit_commitments=tuple(tp["diff_bit_commitments"]),
                diff_bit_proofs=tuple(
                    BitProof(a0=p["a0"], a1=p["a1"], e0=p["e0"], e1=p["e1"], s0=p["s0"], s1=p["s1"])
                    for p in tp["diff_bit_proofs"]
                ),
                n_bits=tp["n_bits"],
            ),
        )


def generate_detection_proof(
    score: float,
    threshold: float,
    record_id: str,
    model_name: str,
    scale: int = 10_000,
    n_bits: int = 16,
) -> DetectionProof:
    """Build a proof that `score >= threshold` for one flagged record.

    `n_bits=16` covers fixed-point values up to 65535 at the default
    `scale=10_000` -- this project's scores live in [0, 1], so fixed-point
    values top out at 10_000, comfortably inside 16 bits with headroom
    to spare, while keeping proof generation/verification to a handful
    of seconds (see tests/test_zkp.py for the measured cost) rather than
    the ~2x-slower cost of the 32-bit width used only for the group-
    primitive smoke test in this package's own development.

    Raises ValueError if `score < threshold` -- there is no proof to
    generate for a record that doesn't actually clear the threshold.
    """
    fixed_score = to_fixed_point(score, scale)
    fixed_threshold = to_fixed_point(threshold, scale)
    commitment, blinding = commit(fixed_score)
    context = _context(record_id, model_name, fixed_threshold, scale, n_bits)
    threshold_proof = prove_at_least_threshold(fixed_score, blinding, fixed_threshold, n_bits, context)
    return DetectionProof(
        record_id=record_id,
        model_name=model_name,
        threshold_fixed_point=fixed_threshold,
        scale=scale,
        n_bits=n_bits,
        value_commitment=commitment.value,
        threshold_proof=threshold_proof,
    )


def verify_detection_proof(proof: DetectionProof) -> bool:
    """Verify a `DetectionProof` entirely from its own public fields --
    no secret material, no access to the original score, required.
    """
    context = _context(
        proof.record_id, proof.model_name, proof.threshold_fixed_point, proof.scale, proof.n_bits
    )
    return verify_at_least_threshold(
        Commitment(proof.value_commitment), proof.threshold_fixed_point, proof.threshold_proof, context
    )
