"""Central reproducibility helper.

Every entrypoint that generates or trains anything should call
`set_global_seed()` exactly once, at the start, and thread the returned
`numpy.random.Generator` through its own functions explicitly rather than
relying on global numpy state.
"""
from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed python's `random`, numpy's legacy global state, and return a
    fresh seeded `numpy.random.Generator` for explicit use.

    Torch is not currently a dependency of this project -- no phase through
    Phase 3 uses a deep-learning component (Isolation Forest, XGBoost, and a
    River online learner are all torch-free). If a torch-based component is
    ever added, seed it here too (`torch.manual_seed(seed)`) and update this
    docstring; until then, omitting it is a documented non-issue rather than
    an oversight.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
