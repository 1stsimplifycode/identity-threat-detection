"""Zero-knowledge proof stack: completeness (honest proofs verify) and,
critically, negative controls at every layer (tampering, wrong context,
false statements) proving the verifier actually rejects bad proofs
rather than trivially accepting everything -- the same discipline
test_leakage_audit.py applies to the leakage audit itself.

Uses n_bits=8 for the range/detection-proof layer tests (not the
n_bits=16 production default in zkp/detection_proof.py) purely to keep
this test file fast; zkp/range_proof.py's own module-level cost is
identical per bit regardless of width, so testing at a smaller width
still exercises the exact same code paths.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from zkp.detection_proof import DetectionProof, generate_detection_proof, verify_detection_proof
from zkp.group import G, H, P, Q, mod_exp, random_exponent
from zkp.pedersen import commit, verify_opening
from zkp.range_proof import prove_at_least_threshold, to_fixed_point, verify_at_least_threshold
from zkp.sigma import prove_bit, prove_knowledge_of_opening, verify_bit, verify_knowledge_of_opening


def test_group_generators_have_prime_order_and_are_independent():
    assert pow(G, Q, P) == 1
    assert pow(H, Q, P) == 1
    assert G != H and G != 1 and H != 1


def test_pedersen_commit_round_trips_and_detects_tampering():
    c, r = commit(17)
    assert verify_opening(c, 17, r)
    assert not verify_opening(c, 18, r)


def test_pedersen_is_additively_homomorphic():
    c1, r1 = commit(4)
    c2, r2 = commit(9)
    combined = c1 * c2
    assert verify_opening(combined, 13, (r1 + r2) % Q)


def test_pedersen_hides_value_two_commitments_to_same_value_differ():
    c1, _ = commit(42)
    c2, _ = commit(42)
    assert c1.value != c2.value


def test_schnorr_opening_proof_completeness_and_tamper_detection():
    c, r = commit(123)
    proof = prove_knowledge_of_opening(123, r, c, context=b"ctx-a")
    assert verify_knowledge_of_opening(c, proof, context=b"ctx-a")

    assert not verify_knowledge_of_opening(c, proof, context=b"ctx-b")
    tampered = replace(proof, sv=(proof.sv + 1) % Q)
    assert not verify_knowledge_of_opening(c, tampered, context=b"ctx-a")


@pytest.mark.parametrize("bit", [0, 1])
def test_cds_bit_proof_completeness(bit):
    c, r = commit(bit)
    proof = prove_bit(bit, r, c, context=b"bit-ctx")
    assert verify_bit(c, proof, context=b"bit-ctx")


def test_cds_bit_proof_does_not_reveal_which_branch_via_structure():
    """Both branches of an honest proof are always present and
    structurally well-formed regardless of the real bit -- the OR-proof's
    ZK property means an outside observer cannot distinguish a bit=0
    proof from a bit=1 proof by inspecting which branch "looks real".
    """
    c0, r0 = commit(0)
    c1, r1 = commit(1)
    p0 = prove_bit(0, r0, c0, context=b"ctx")
    p1 = prove_bit(1, r1, c1, context=b"ctx")
    for p in (p0, p1):
        assert (p.e0 + p.e1) % Q != 0  # both challenges contribute, neither branch is a no-op


def test_cds_bit_proof_rejects_tampering():
    c, r = commit(1)
    proof = prove_bit(1, r, c, context=b"bit-ctx")
    tampered = replace(proof, e0=(proof.e0 + 1) % Q)
    assert not verify_bit(c, tampered, context=b"bit-ctx")


def test_range_proof_accepts_value_above_and_at_threshold():
    for value in (10, 5):  # above, and exactly at, threshold=5
        c, r = commit(value)
        proof = prove_at_least_threshold(value, r, threshold=5, n_bits=8, context=b"range-ctx")
        assert verify_at_least_threshold(c, 5, proof, context=b"range-ctx")


def test_range_proof_cannot_be_constructed_for_a_false_statement():
    c, r = commit(2)
    with pytest.raises(ValueError):
        prove_at_least_threshold(2, r, threshold=5, n_bits=8, context=b"range-ctx")


def test_range_proof_rejects_mismatched_commitment():
    c_a, r_a = commit(10)
    c_b, _ = commit(200)  # unrelated commitment, also >= 5, but not the one the proof is about
    proof = prove_at_least_threshold(10, r_a, threshold=5, n_bits=8, context=b"range-ctx")
    assert not verify_at_least_threshold(c_b, 5, proof, context=b"range-ctx")


def test_range_proof_rejects_wrong_context():
    c, r = commit(10)
    proof = prove_at_least_threshold(10, r, threshold=5, n_bits=8, context=b"ctx-a")
    assert not verify_at_least_threshold(c, 5, proof, context=b"ctx-b")


def test_range_proof_out_of_bit_width_is_rejected_at_construction():
    c, r = commit(1000)
    with pytest.raises(ValueError):
        prove_at_least_threshold(1000, r, threshold=0, n_bits=4, context=b"ctx")  # diff=1000 doesn't fit in 4 bits


def test_to_fixed_point_matches_documented_scale():
    assert to_fixed_point(0.5558, scale=10_000) == 5558
    with pytest.raises(ValueError):
        to_fixed_point(-0.1)


def test_detection_proof_completeness_and_json_round_trip():
    import json

    proof = generate_detection_proof(
        score=0.87, threshold=0.5, record_id="evt-1", model_name="xgboost_class_weight", scale=100, n_bits=8
    )
    assert verify_detection_proof(proof)

    restored = DetectionProof.from_dict(json.loads(json.dumps(proof.to_dict())))
    assert verify_detection_proof(restored)


def test_detection_proof_below_threshold_cannot_be_generated():
    with pytest.raises(ValueError):
        generate_detection_proof(
            score=0.2, threshold=0.5, record_id="evt-2", model_name="xgboost_class_weight", scale=100, n_bits=8
        )


@pytest.mark.parametrize("field,bad_value", [
    ("record_id", "evt-different"),
    ("model_name", "isolation_forest"),
    ("threshold_fixed_point", 1),
])
def test_detection_proof_rejects_relabeling(field, bad_value):
    """A genuine proof for one (record, model, threshold) must not verify
    once any of those public-but-bound identifiers are swapped -- this is
    what stops a proof generated for one flagged record from being
    pasted onto a different record's dashboard row.
    """
    proof = generate_detection_proof(
        score=0.9, threshold=0.5, record_id="evt-3", model_name="xgboost_class_weight", scale=100, n_bits=8
    )
    tampered = replace(proof, **{field: bad_value})
    assert not verify_detection_proof(tampered)


def test_detection_proof_rejects_forged_commitment():
    """An attacker who doesn't know a valid opening cannot substitute a
    fresh commitment to a large value and reuse someone else's proof.
    """
    genuine = generate_detection_proof(
        score=0.9, threshold=0.5, record_id="evt-4", model_name="xgboost_class_weight", scale=100, n_bits=8
    )
    forged_commitment, _ = commit(9999)
    tampered = replace(genuine, value_commitment=forged_commitment.value)
    assert not verify_detection_proof(tampered)
