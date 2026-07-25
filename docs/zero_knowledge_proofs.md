# Zero-knowledge proofs

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.**

> **Prototype cryptography, not an audited library.** Everything in `zkp/`
> is a hand-rolled implementation of textbook (Pedersen / Sigma-protocol /
> Cramer-Damgård-Schoenmakers) constructions, written to demonstrate the
> technique end-to-end and pass its own correctness/soundness test suite
> (`tests/test_zkp.py`). It is appropriate for exactly that -- a documented
> prototype -- and is **not** a substitute for an audited library (e.g. a
> maintained Bulletproofs or Groth16 implementation) before any real
> deployment on data that matters. See "Threat model and honest limits"
> below for the specific things this does and does not protect against.

## What this proves, in one sentence

*"This flagged record's anomaly score is at or above the model's operating
threshold" -- provable and independently verifiable without the proof
itself ever revealing the score.*

## Why: two purposes, both in scope

The project's evaluation pipeline (`docs/phase_3_evaluation_report.md`) and
dashboard already establish *that* a model flags certain events. This
feature adds a cryptographic layer on top, for two purposes:

1. **Detection integrity.** A flagged anomaly's "it crossed the threshold"
   claim is bound to a specific `record_id`, `model_name`, and threshold
   value, and can be checked by anyone holding the proof -- without
   trusting the dashboard's own rendering code, and without needing
   access to the underlying feature values or model weights.
2. **Analyst-facing privacy.** The verification step never requires
   opening the commitment (i.e. never requires learning the score). A
   viewer with only the proof, not the score, can confirm "yes, this
   crossed the threshold" and nothing more.

## Threat model and honest limits

**What is proven:** given a Pedersen commitment `C` to some integer
`v` (a fixed-point-scaled anomaly score) and a public threshold `t`,
that `v >= t`, without revealing `v`. The proof is bound (via
Fiat-Shamir's hashed context, see below) to a specific `record_id`,
`model_name`, and `t`, so a genuine proof for one record cannot be
replayed as if it were about a different record, a different model, or
a different threshold.

**What is NOT proven -- the zkML boundary.** This does **not** prove
that the committed score `v` was itself correctly computed by the
deployed model from the record's actual input features. Doing that
would require a *zkML* system -- a zero-knowledge circuit over the
model's own computation graph (e.g. EZKL compiling an ONNX model to a
Halo2 circuit) -- which is a substantially larger undertaking (compiling
gradient-boosted trees or a fine-tuned transformer to an arithmetic
circuit, then proving/verifying that circuit's execution) and is
explicitly out of scope for this prototype. In plain terms: this proves
"the threshold decision, given the score, was correct and belongs to
this record" -- not "the score was honestly computed from the record's
real behavioral data." A deployment that needed the stronger guarantee
would need to add a zkML layer on top of (not instead of) what's here.

**What "detection integrity" concretely buys, given that boundary:** it
stops a specific, narrower class of tampering -- someone altering a
score *after* it was committed (e.g. editing a dashboard artifact file
to inflate or suppress which records show as flagged) -- because doing
so breaks the proof. It does not stop a dishonest *prover* (someone who
controls the offline `prepare_data.py` step) from committing to a score
their model didn't actually produce; that requires the prover to be
trusted, exactly as it is everywhere else in this pipeline (nothing here
changes who is trusted to run the offline model-scoring step).

**Prototype-grade crypto, stated plainly:** hand-rolled, not
security-audited. The group parameters (RFC 3526 "MODP Group 14", a
standardized, decades-reviewed 2048-bit safe prime) are the one piece
borrowed from an external, audited source rather than generated locally
-- see `zkp/group.py`'s module docstring for why. Everything built on
top of that group (Pedersen commitments, Schnorr proofs, the CDS
OR-proof, the bit-decomposition range proof) is this project's own
implementation of well-known constructions, verified by direct test
execution (`tests/test_zkp.py`), not by external cryptographic review.

## How it works

### 1. Fixed-point encoding (`zkp/range_proof.py::to_fixed_point`)

Model scores in this project are floats in `[0, 1]`. The proof system
works over integers, so a score is scaled and rounded:
`fixed = round(score * scale)`, with `scale = 10_000` by default (4
decimal digits -- matching the precision already reported in the
evaluation tables, e.g. `precision: 0.5558`). No information the
dashboard doesn't already display elsewhere is lost by this rounding.

### 2. Pedersen commitment (`zkp/pedersen.py`)

`C = G^value * H^blinding mod P`, over a 2048-bit safe-prime group
(`zkp/group.py`). Two properties matter:

- **Perfectly hiding:** `C` alone leaks nothing about `value` -- every
  possible value has *some* blinding factor that opens `C` to it.
- **Computationally binding:** the committer cannot later claim `C`
  opens to a different value than the one actually committed, without
  solving a discrete-log problem in this group (believed intractable).

Commitments are additively homomorphic: `commit(a) * commit(b) ==
commit(a + b)` (mod arithmetic) -- the range proof below depends on
this directly.

### 3. Sigma protocols, made non-interactive via Fiat-Shamir (`zkp/sigma.py`)

A **Schnorr proof-of-knowledge-of-opening** lets a prover show they know
*some* `(value, blinding)` opening a commitment, without revealing it.
A **Cramer-Damgård-Schoenmakers (1994) OR-proof** extends this to prove
a committed bit is 0 *or* 1, without revealing which -- built by proving
two Schnorr-style statements where only the true one is computed
honestly and the other is simulated, then combined so a verifier cannot
tell which branch was real. **Fiat-Shamir** replaces the interactive
protocol's random verifier challenge with a SHA-256 hash of the
prover's own first message plus a `context` byte-string, letting the
prover compute the entire proof once, offline -- exactly what letting
`prepare_data.py` generate proofs once and `app.py` verify them on
every page load requires, with no live prover involved at verification
time.

### 4. Bit-decomposition range proof (`zkp/range_proof.py`)

To prove `value >= threshold`:

1. Compute `diff = value - threshold`. If `diff < 0`, there is no valid
   proof to construct -- the claim is false, full stop.
2. Commit to each bit of `diff` (up to a fixed public `n_bits` width)
   separately, and prove each is 0-or-1 via the CDS OR-proof.
3. The product of the bit commitments raised to their place values
   (`prod(C_i ** 2**i)`) is, by the homomorphic property, itself a
   commitment to `diff` with a *derived* blinding factor. The prover
   solves for the last bit's blinding so this derived commitment's
   blinding matches the original commitment's blinding exactly.
4. The verifier checks every per-bit OR-proof, then checks that the
   recombined bit commitments, multiplied by a (public, zero-blinding)
   commitment to `threshold`, equal the original value commitment. If
   both hold, `diff` is guaranteed non-negative and `< 2**n_bits`, which
   means `value >= threshold` -- without the verifier ever learning
   `value`, `diff`, or any individual bit.

This is a standard (simplified, unoptimized) range-proof construction:
proof size and cost are `O(n_bits)`. No aggregation tricks (e.g.
Bulletproofs' logarithmic-size proofs) are used -- appropriate at this
project's scale (see "Performance" below), not appropriate if this were
scaled to proving every event in a full production stream.

### 5. Binding to a detection record (`zkp/detection_proof.py`)

The application layer folds `record_id`, `model_name`, and the
fixed-point `threshold` into the Fiat-Shamir `context` (alongside a
fixed domain-separation prefix), so a genuine proof cannot be replayed
against a different record, a different model, or a different
threshold. `DetectionProof.to_dict()` / `.from_dict()` round-trip
through the JSON artifact `prepare_data.py` writes and `app.py` reads.

## Where the proofs live

- **Generation:** `dashboard/prepare_data.py`, offline, alongside every
  other precomputed dashboard artifact (SHAP, streaming-approx,
  feature history). Config: `configs/models/default.yaml`'s `zkp:`
  block.
- **Storage:** `dashboard/data/<run_name>/zkp_proofs.json` -- a flat
  `{record_id: proof}` map plus the shared `model_name`, `threshold`,
  `scale`, and `n_bits` used for the whole batch.
- **Verification:** `dashboard/app.py`, at page-render time, per
  selected event -- `zkp/detection_proof.py::verify_detection_proof`.
  Pure Python stdlib (`hashlib`, `secrets`, `dataclasses`); no new entry
  needed in `requirements-dashboard.txt` (see that file's own comment).

## Why only a SAMPLE of flagged events get a proof

Each proof takes a few seconds to generate and a few seconds to verify
(measured directly, see "Performance" below) -- orders of magnitude
slower than everything else in this pipeline, because it's genuine
2048-bit-group modular exponentiation, run tens of times per proof (once
per bit). Generating one for every flagged event would be fine at
`small_dev` scale but would take **hours** at the full evaluation's
scale (thousands of flagged events per model, per
`docs/phase_3_evaluation_report.md`'s `n_flagged` column) for a feature
whose purpose here is to demonstrate the cryptographic technique
end-to-end, not to cryptographically cover every row. `configs/models/
default.yaml`'s `zkp.max_proof_samples` (default: 25) caps this,
mirroring the existing `shap_explainability.max_flagged_samples` pattern
for exactly the same reason. Events outside the sample show a plain
"not in this run's sampled proof set" caption in the dashboard --
stated honestly, not silently omitted.

## What this demonstrates vs. what it would gate

This build's dashboard (`dashboard/app.py`) is a **full-access internal
analyst tool** -- it already shows the raw `score` column in the flagged
events table, by existing design (the live threshold slider needs it).
The zero-knowledge proof badge added alongside an event's detail view is
therefore a **demonstration of the mechanism**, verified independently
of that visible score column, not something currently gating access to
a hidden score in this particular build. A deployment that actually
needed analyst-facing privacy (e.g. a redacted view for a lower-trust
viewer, a compliance auditor, or an external stakeholder) would reuse
the exact same `zkp_proofs.json` artifact and `verify_detection_proof`
call, just in a view that never sends that viewer the `score` column at
all -- the proof already works without it; only the surrounding UI
would need to change.

## Performance (measured directly, not estimated)

At the default config (`scale=10_000`, `n_bits=16`):

| Operation | Time |
|---|---|
| Group setup (primality self-checks, one-time per process) | ~1.2s |
| Proof generation (one record) | ~3.2s |
| Proof verification (one record) | ~3.6s |
| Serialized proof size (JSON) | ~70 KB |

At `n_bits=32` (tested during development, not the shipped default):
~6.2s generate / ~6.3s verify -- roughly double, as expected for a
linear-in-`n_bits` construction. `n_bits=16` was chosen because scores
scaled by `scale=10_000` never exceed `10_000`, comfortably inside 16
bits (max 65,535) with headroom, at half the cost of the 32-bit width.

At the default `zkp.max_proof_samples=25`, one `prepare_data.py` run
adds roughly 25 x ~3.2s = ~80 seconds of offline proof generation --
negligible next to the rest of the pipeline (feature computation, model
training) that step already does.

## Verifying the claims in this document yourself

```bash
# Full correctness + soundness test suite (completeness, tamper-detection,
# and false-statement rejection at every layer: group, Pedersen, Sigma
# protocols, range proof, and the detection-proof application layer).
pytest tests/test_zkp.py -v

# Generate dashboard artifacts including zkp_proofs.json for small_dev.
python -m dashboard.prepare_data --config-name small_dev

# Then run the dashboard locally and select a flagged event with a proof
# to see the "Cryptographic verification" section render live.
streamlit run dashboard/app.py
```
