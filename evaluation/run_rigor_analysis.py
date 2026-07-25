"""Phase 5b (evaluation rigor) entrypoint: calibration, bootstrap
confidence intervals, balanced accuracy, and significance testing between
the three XGBoost imbalance-handling variants.

Deliberately reads the model suite's EXISTING checkpoints
(`data/runs/<run>/checkpoints/*_eval.pkl` for each model's chosen operating
`threshold`, `*_scores.parquet` for its test-set scores) rather than
calling `build_model_suite()` -- every statistic here is a pure function of
already-computed (y_true, y_score, y_pred), so this adds real, reproducible
rigor on top of Phase 5's retrain WITHOUT retraining anything again. If the
checkpoints don't exist yet, run `evaluation.run_evaluation` or
`dashboard.prepare_data` first.

Usage:
    python -m evaluation.run_rigor_analysis --config-name small_dev
"""
from __future__ import annotations

import pickle
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score

from evaluation.calibration import bootstrap_metric_ci, compute_calibration, paired_bootstrap_significance
from evaluation.chronological_split import apply_split, chronological_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Only genuinely probabilistic [0, 1] scores get calibration analysis --
# rule_based_baseline's score is a raw count of 3 named flags (0-3, not a
# probability) and isolation_forest's is an unbounded signed anomaly score;
# reporting Brier score/ECE for either would imply a meaning the score was
# never designed to have. Verified by direct inspection of each model's
# score range (see docs/phase_5b_evaluation_rigor.md).
CALIBRATABLE_MODELS = ["xgboost_none", "xgboost_class_weight", "xgboost_smote", "river_online", "hf_bert_tiny"]
ALL_MODELS = ["rule_based_baseline", "isolation_forest"] + CALIBRATABLE_MODELS
XGBOOST_VARIANTS = ["xgboost_none", "xgboost_class_weight", "xgboost_smote"]


def _load_base_test(run_dir: Path, cfg: DictConfig) -> pd.DataFrame:
    events = pd.read_parquet(run_dir / "events.parquet")
    labels = pd.read_parquet(run_dir / "labels.parquet")
    split = chronological_split(events, train_frac=float(cfg.evaluation.train_frac))
    labels_slim = labels[["record_id", "is_attack", "attack_type", "attack_id"]]
    events_slim = events[["record_id", "timestamp"]]
    return apply_split(events_slim, split, "test").merge(labels_slim, on="record_id", how="left")


def _render_report(
    calibration_rows: list[dict], ci_rows: list[dict], significance_rows: list[dict],
) -> str:
    lines = [
        "# Phase 5b: Evaluation Rigor (calibration, confidence intervals, significance)",
        "",
        "> Synthetic data. Not derived from or validated against real organizational logs. "
        "For benchmarking detection methods only.",
        "",
        "Post-hoc statistics computed directly over the model suite's existing test-set "
        "checkpoints -- no retraining. Every bootstrap uses a fixed seed (42); re-running "
        "this exact command reproduces these numbers exactly.",
        "",
        "## Balanced accuracy + calibration",
        "",
        "`balanced_accuracy` is reported for every model (valid regardless of score type). "
        "Brier score and Expected Calibration Error (ECE) are reported only for models whose "
        "score is a genuine [0, 1] probability -- `rule_based_baseline` (a 0-3 flag count) and "
        "`isolation_forest` (an unbounded signed anomaly score) are excluded, not silently "
        "given a misleading number.",
        "",
        pd.DataFrame(calibration_rows).to_markdown(index=False),
        "",
        "## Bootstrap 95% confidence intervals (300 resamples, seed=42)",
        "",
        "Nonparametric percentile bootstrap over the test-set indices -- how much each metric "
        "would plausibly vary on a different sample from the same underlying population, not "
        "just a single point estimate presented as exact.",
        "",
        pd.DataFrame(ci_rows).to_markdown(index=False),
        "",
        "## Significance: are the 3 XGBoost imbalance methods really different? (PR-AUC, paired bootstrap, 300 resamples)",
        "",
        "Paired bootstrap on PR-AUC (the problem statement's headline metric under imbalance) "
        "-- both models scored on the SAME resampled test rows each iteration. p < 0.05 means "
        "the gap is unlikely to be bootstrap noise; a 95% CI on the difference that excludes 0 "
        "says the same thing from the interval side.",
        "",
        pd.DataFrame(significance_rows).to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../configs", config_name="small_dev")
def main(cfg: DictConfig) -> None:
    run_dir = PROJECT_ROOT / cfg.run.output_dir
    checkpoint_dir = run_dir / "checkpoints"
    base_test = _load_base_test(run_dir, cfg)

    scores_by_model: dict[str, pd.DataFrame] = {}
    calibration_rows: list[dict] = []
    ci_rows: list[dict] = []
    reliability_bin_frames: list[pd.DataFrame] = []

    for name in ALL_MODELS:
        eval_path = checkpoint_dir / f"{name}_eval.pkl"
        scores_path = checkpoint_dir / f"{name}_scores.parquet"
        if not (eval_path.exists() and scores_path.exists()):
            print(f"skipping {name}: no checkpoint found (run evaluation.run_evaluation or dashboard.prepare_data first)")
            continue

        with open(eval_path, "rb") as f:
            ev = pickle.load(f)
        scores_df = pd.read_parquet(scores_path)
        frame = base_test.merge(scores_df, on="record_id", how="left")
        y_true = frame["is_attack"].astype(int).to_numpy()
        y_score = frame["score"].to_numpy()
        y_pred = (y_score >= ev.threshold).astype(int)
        scores_by_model[name] = frame.set_index("record_id")["score"]

        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        row = {"model": name, "balanced_accuracy": round(bal_acc, 4)}
        if name in CALIBRATABLE_MODELS:
            calib = compute_calibration(name, y_true, y_score)
            row["brier_score"] = round(calib.brier_score, 4)
            row["ece"] = round(calib.ece, 4)
            bins = calib.reliability_bins.copy()
            bins.insert(0, "model", name)
            reliability_bin_frames.append(bins)
        else:
            row["brier_score"] = None
            row["ece"] = None
        calibration_rows.append(row)
        print(f"computed calibration/balanced-accuracy: {name}")

        ci = bootstrap_metric_ci(y_true, y_score, y_pred)
        for metric_name, (point, lo, hi) in ci.items():
            ci_rows.append({
                "model": name, "metric": metric_name,
                "estimate": round(point, 4), "ci_95_lo": round(lo, 4), "ci_95_hi": round(hi, 4),
            })
        print(f"computed bootstrap CIs: {name}")

    significance_rows: list[dict] = []
    y_true_full = base_test["is_attack"].astype(int)
    for i, a in enumerate(XGBOOST_VARIANTS):
        for b in XGBOOST_VARIANTS[i + 1:]:
            if a not in scores_by_model or b not in scores_by_model:
                continue
            common_ids = scores_by_model[a].index.intersection(scores_by_model[b].index)
            yt = base_test.set_index("record_id").loc[common_ids, "is_attack"].astype(int).to_numpy()
            sa = scores_by_model[a].loc[common_ids].to_numpy()
            sb = scores_by_model[b].loc[common_ids].to_numpy()
            result = paired_bootstrap_significance(a, b, yt, sa, sb, metric="pr_auc")
            significance_rows.append({
                "model_a": result.model_a, "model_b": result.model_b,
                "pr_auc_a": round(result.value_a, 4), "pr_auc_b": round(result.value_b, 4),
                "diff": round(result.diff, 4),
                "diff_95_ci": f"[{result.diff_ci_lo:.4f}, {result.diff_ci_hi:.4f}]",
                "p_value": round(result.p_value, 4),
            })
            print(f"computed significance: {a} vs {b}")

    report_md = _render_report(calibration_rows, ci_rows, significance_rows)
    out_path = PROJECT_ROOT / "docs" / "phase_5b_evaluation_rigor.md"
    out_path.write_text(report_md, encoding="utf-8")
    print(f"\nWrote report to {out_path}")

    # -- dashboard artifacts: same real numbers as the markdown report
    # above, just also in parquet form so dashboard/app.py's Calibration
    # section (Phase 5d) can render them without recomputing anything. --
    dash_out_dir = PROJECT_ROOT / "dashboard" / "data" / str(cfg.run.name)
    dash_out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calibration_rows).to_parquet(dash_out_dir / "calibration_summary.parquet", index=False)
    pd.DataFrame(ci_rows).to_parquet(dash_out_dir / "bootstrap_ci.parquet", index=False)
    pd.DataFrame(significance_rows).to_parquet(dash_out_dir / "significance.parquet", index=False)
    if reliability_bin_frames:
        pd.concat(reliability_bin_frames, ignore_index=True).to_parquet(dash_out_dir / "calibration_bins.parquet", index=False)
    print(f"Wrote dashboard calibration artifacts to {dash_out_dir}")


if __name__ == "__main__":
    main()
