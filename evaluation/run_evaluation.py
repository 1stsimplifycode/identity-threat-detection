"""Phase 3 evaluation entrypoint: builds the full model suite (see
`evaluation/model_suite.py`) and renders the six-criteria report.

Usage:
    python -m evaluation.run_evaluation --config-name small_dev
"""
from __future__ import annotations

# IMPORTANT: torch must be imported before pandas/pyarrow anywhere in this
# process -- see models/README.md for the full explanation (a genuine
# Windows DLL conflict between pyarrow's bundled Arrow C++ runtime and
# torch's bundled libraries, found by direct testing during Phase 3).
import torch  # noqa: F401

from pathlib import Path

import hydra
from omegaconf import DictConfig

from evaluation.model_suite import build_model_suite
from evaluation.report import attack_type_recall_table, comparison_table, render_markdown_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@hydra.main(version_base=None, config_path="../configs", config_name="small_dev")
def main(cfg: DictConfig) -> None:
    run_dir = PROJECT_ROOT / cfg.run.output_dir
    include_hf = bool(cfg.evaluation.get("include_hf", True))
    # Checkpointed by default: at full (Scale-up) scale a run can take well
    # over an hour, and this makes an interrupted run resumable (re-running
    # this exact command loads already-finished stages from
    # data/runs/<run>/checkpoints/ instead of redoing them) -- see
    # evaluation/model_suite.py's module docstring and docs/scale_up_report.md.
    checkpoint_dir = run_dir / "checkpoints"
    print(f"Checkpointing to {checkpoint_dir} -- re-running this command resumes from here if interrupted.")
    suite = build_model_suite(run_dir, cfg, include_hf=include_hf, checkpoint_dir=checkpoint_dir)
    if not include_hf:
        print("NOTE: hf_bert_tiny skipped (evaluation.include_hf=false) -- "
              "not included in this run's comparison table. See docs/deployment.md.")

    comp_table = comparison_table(suite.evaluations)
    attack_table = attack_type_recall_table(suite.evaluations)
    print(f"Computed {len(suite.features)} feature rows in {suite.feature_elapsed_seconds:.1f}s "
          f"({len(suite.features) / suite.feature_elapsed_seconds:.0f} events/sec)")
    print(suite.split.summary())
    for ev in suite.evaluations:
        print(f"evaluated: {ev.name}")
    print()
    print(comp_table.to_string(index=False))

    scalability = {
        "events_per_sec": len(suite.features) / suite.feature_elapsed_seconds,
        "feature_compute_seconds": suite.feature_elapsed_seconds,
        "n_events": len(suite.features),
    }
    # getattr, not ev.class_thresholds: a checkpoint pickled before this field
    # existed restores an object missing the attribute entirely (pickle
    # restores __dict__ directly, bypassing the dataclass default).
    class_thresholds = {
        ev.name: getattr(ev, "class_thresholds", {}) for ev in suite.evaluations if getattr(ev, "class_thresholds", {})
    }
    report_md = render_markdown_report(
        comp_table, attack_table, scalability, suite.split.summary(), suite.multiclass_reports, class_thresholds,
    )

    print()
    print("ADWIN drift detection vs. ground truth:")
    print(suite.drift_eval.to_string(index=False))

    report_md += "\n## Drift detection (ADWIN vs. ground-truth drift_log.csv)\n\n"
    report_md += suite.drift_eval.to_markdown(index=False) + "\n"

    out_path = PROJECT_ROOT / "docs" / "phase_3_evaluation_report.md"
    out_path.write_text(report_md, encoding="utf-8")
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
