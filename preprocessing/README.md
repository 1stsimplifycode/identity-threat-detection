# preprocessing/

Shared, low-level building blocks used by every other package:

- **`constants.py`** -- canonical categorical vocabularies (departments,
  privilege levels, auth methods, ...). Single source of truth so generation
  code and schema validation can never silently disagree on what values are
  allowed.
- **`geo_utils.py`** -- haversine distance and the synthetic city pools used
  for home locations, legitimate travel, and attack destinations.
- **`schema.py`** -- one schema dict per table (`USERS_SCHEMA`,
  `EVENTS_SCHEMA`, `LABELS_SCHEMA`, `ATTACKS_SCHEMA`) plus
  `validate_dataframe()`, which raises a single `ValueError` listing every
  violation found.

## Phase status

Phase 1: schema + validation for the 4 core tables (users, events, labels,
attacks). Real-time/streaming preprocessing (as opposed to batch schema
validation) is not part of this package -- see `feature_engineering/` for the
dual batch/streaming feature interface introduced in Phase 2.
