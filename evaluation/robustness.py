"""Phase 5c (robustness / stress testing): perturbation functions used to
stress-test an ALREADY-TRAINED model's fixed decision boundary against
noisy or missing real-world inputs. Deliberately narrow in scope --
additive Gaussian noise scaled to each feature's own train-set std, and
group-level ablation (replace with the train-set median, a realistic
"this feature was unavailable, imputed with a typical value" scenario) --
not full adversarial/gradient-based evasion, which is a materially
different undertaking that would need explicit scoping first (see
docs/phase_5c_robustness.md's "Not attempted" section).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject_gaussian_noise(
    features: pd.DataFrame, feature_columns: list[str], feature_stds: pd.Series,
    noise_fraction: float, seed: int,
) -> pd.DataFrame:
    """Adds N(0, (noise_fraction * feature_std)^2) to each feature column
    independently. `feature_stds` MUST come from the TRAIN split (passing
    test-set stds would leak test distribution information into the
    perturbation itself). `noise_fraction=0` returns an identical copy --
    the baseline / no-noise control.
    """
    rng = np.random.default_rng(seed)
    out = features.copy()
    for col in feature_columns:
        std = float(feature_stds[col])
        if std == 0.0 or noise_fraction == 0.0:
            continue
        out[col] = out[col].to_numpy() + rng.normal(0.0, noise_fraction * std, size=len(out))
    return out


def ablate_features(features: pd.DataFrame, columns_to_ablate: list[str], train_medians: pd.Series) -> pd.DataFrame:
    """Replaces the given columns with their TRAIN-split median -- simulates
    "this feature was unavailable at inference time and got imputed with a
    typical value," a realistic missing-feature scenario, rather than
    zeroing (which is often itself an out-of-distribution, unrealistic
    value for a feature that's never actually 0 in practice).
    """
    out = features.copy()
    for col in columns_to_ablate:
        out[col] = float(train_medians[col])
    return out
