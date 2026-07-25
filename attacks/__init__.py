"""Attack generators and injection orchestration.

Phase 1 implements two attack types: brute_force and impossible_travel.
Every attack type must have a corresponding entry in
configs/mitre_mapping.yaml *before* its generator is written -- see
attacks/base.py:load_mitre_mapping.
"""
