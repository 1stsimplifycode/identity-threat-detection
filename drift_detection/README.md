# drift_detection/

- **`adwin_detector.py`** (Phase 3) -- `river.drift.ADWIN`, monitoring
  `geo_distance_from_home_km` (computed against each user's ORIGINAL home
  coordinates, which never update on relocation -- so a real drift shows up
  as a genuine, sustained shift). Evaluated against the ground-truth
  `data/runs/<run>/drift_log.csv` from `generator/drift.py`.

## A real finding, not just a working number

Feeding raw per-event values (or a rolling mean over raw event order) into
ADWIN fired **hundreds of times across the entire stream**, including well
before the configured drift day -- not a meaningful detection, just noise
from mixing many different users' individual baseline jitter. A clean
**daily mean** showed an unmistakable step change exactly at the drift day
(confirming the injection itself works), but with too few days for ADWIN to
build statistical confidence before the stream ends. The fix that actually
worked: aggregating by a **fixed event count per bin** (not calendar day),
which gives both a clean signal and enough bins. See the module's own
docstring for the full investigation. Result on real `small_dev` data:
detection lag 0.49 days after the configured drift day.
