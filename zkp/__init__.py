"""Zero-knowledge proofs for detection integrity and analyst-facing privacy.

See docs/zero_knowledge_proofs.md for the threat model, protocol design, and
honest limitations. Short version: this proves "a committed anomaly score
exceeds the public decision threshold" without revealing the score, using
Pedersen commitments and Sigma-protocol range proofs (Fiat-Shamir,
non-interactive) over a standardized RFC 3526 safe-prime group -- not a
zk-SNARK, and it does NOT prove the score was correctly computed by the
model (that would require a zkML circuit, out of scope here; see the docs
for what "detection integrity" actually covers).
"""
