#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Large / forecast-output fields that we explicitly do NOT want to flatten.
SKIP_KEYS = {
    "terminal_samples",
    "path_mean",
    "path_median",
    "path_lower",
    "path_upper",
    "intervals",
    "forecast_dates",
    "train_dates",
    "prices",
}


def is_scalar(x):
    return (
        x is None
        or isinstance(x, (str, bool, int, float, np.integer, np.floating))
    )


def flatten_scalars(d, prefix="", out=None):
    """
    Recursively flatten scalar values from a dictionary.

    Lists/arrays are ignored except for convergence diagnostics handled separately.
    """
    if out is None:
        out = {}

    if not isinstance(d, dict):
        return out

    for key, value in d.items():
        if key in SKIP_KEYS:
            continue

        name = f"{prefix}{key}" if prefix else key

        if is_scalar(value):
            out[name] = value

        elif isinstance(value, dict):
            flatten_scalars(value, prefix=name + "__", out=out)

    return out


def summarize_array(x, prefix, out):
    """Store useful summaries of diagnostic arrays."""
    if not isinstance(x, list) or len(x) == 0:
        return

    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return

    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return

    out[f"{prefix}__min"] = float(np.min(arr))
    out[f"{prefix}__max"] = float(np.max(arr))
    out[f"{prefix}__median"] = float(np.median(arr))


def extract_record(path):
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception as exc:
        return {
            "file": str(path),
            "parse_error": repr(exc),
        }

    rec = obj.get("record", obj)

    row = {
        "file": str(path),
        "unit_id": rec.get("unit_id"),
        "ticker": rec.get("target_ticker"),
        "origin": rec.get("origin"),
        "model": rec.get("model"),
        "lookback": rec.get("lookback"),
        "status": rec.get("status"),
        "runtime_seconds": rec.get("runtime_seconds"),
        "starting_price": rec.get("starting_price"),
        "training_return": rec.get("training_return"),
        "training_volatility": rec.get("training_volatility"),
    }

    prediction = rec.get("prediction") or {}
    diagnostics = prediction.get("diagnostics") or {}

    # ------------------------------------------------------------
    # Generic scalar diagnostic parameters
    # ------------------------------------------------------------
    flattened = flatten_scalars(diagnostics, prefix="diag__")
    row.update(flattened)

    # ------------------------------------------------------------
    # Posterior hyperparameters
    #
    # Example:
    #
    # diagnostics["hyperparameters"]["A_mu"]["median"]
    #
    # becomes
    #
    # A_mu__median
    # A_mu__lower68
    # ...
    # ------------------------------------------------------------
    hypers = diagnostics.get("hyperparameters", {})

    if isinstance(hypers, dict):
        for parameter, summary in hypers.items():

            if isinstance(summary, dict):
                for stat, value in summary.items():
                    if is_scalar(value):
                        row[f"{parameter}__{stat}"] = value

            elif is_scalar(summary):
                row[parameter] = summary

    # ------------------------------------------------------------
    # Prior-edge masses
    # ------------------------------------------------------------
    edge = diagnostics.get("prior_edge_mass", {})

    if isinstance(edge, dict):
        for parameter, summary in edge.items():
            if isinstance(summary, dict):
                row[f"{parameter}__edge_lower"] = summary.get("lower")
                row[f"{parameter}__edge_upper"] = summary.get("upper")

    # ------------------------------------------------------------
    # MCMC convergence summaries
    # ------------------------------------------------------------
    rhat = diagnostics.get("split_rhat")
    ess = diagnostics.get("autocorrelation_ess")

    summarize_array(rhat, "rhat", row)
    summarize_array(ess, "ess", row)

    # Convenience aliases
    if isinstance(rhat, list) and len(rhat):
        try:
            row["max_rhat"] = float(np.nanmax(np.asarray(rhat, dtype=float)))
        except Exception:
            pass

    if isinstance(ess, list) and len(ess):
        try:
            row["min_ess"] = float(np.nanmin(np.asarray(ess, dtype=float)))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Useful predictive quantities
    # ------------------------------------------------------------
    for key in [
        "expected_log_return",
        "median_log_return",
        "normal_location",
        "normal_sigma",
        "probability_up",
        "terminal_sigma",
        "terminal_price_median",
        "terminal_price_lower95",
        "terminal_price_upper95",
    ]:
        value = prediction.get(key)
        if is_scalar(value):
            row[f"pred__{key}"] = value

    # ------------------------------------------------------------
    # Realized result, useful later for correlation with parameters
    # ------------------------------------------------------------
    realized = rec.get("realized") or {}

    for key in [
        "terminal_log_return",
        "direction",
        "positive",
        "horizon_realized_volatility",
    ]:
        value = realized.get(key)
        if is_scalar(value):
            row[f"realized__{key}"] = value

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="historical_validation/AMZN/2016-2026-v1",
        help="Historical validation output directory",
    )

    parser.add_argument(
        "--output",
        default="historical_validation/AMZN/2016-2026-v1/parameter_history.csv",
    )

    parser.add_argument(
        "--gp-output",
        default="historical_validation/AMZN/2016-2026-v1/gp_parameter_history.csv",
    )

    args = parser.parse_args()

    root = Path(args.input)

    # Find all JSON checkpoint/record files recursively.
    paths = sorted(root.rglob("*.json"))

    rows = []

    for path in paths:
        row = extract_record(path)

        # Only retain actual forecast records.
        if row.get("origin") is None or row.get("model") is None:
            continue

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No forecast records found below {root}")

    df = pd.DataFrame(rows)

    df["origin"] = pd.to_datetime(df["origin"])

    df = df.sort_values(
        ["origin", "lookback", "model"],
        kind="stable",
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.output, index=False)

    print()
    print("=" * 80)
    print("ALL MODEL PARAMETERS")
    print("=" * 80)
    print(f"records: {len(df)}")
    print(f"origins: {df['origin'].nunique()}")
    print(f"models:  {sorted(df['model'].dropna().unique())}")
    print(f"saved:   {args.output}")

    # ------------------------------------------------------------
    # Focused GP table for our time-evolution study
    # ------------------------------------------------------------

    desired = [
        "origin",
        "model",
        "lookback",
        "status",

        "diag__converged",
        "max_rhat",
        "min_ess",

        "A_mu__median",
        "A_mu__lower68",
        "A_mu__upper68",
        "A_mu__lower95",
        "A_mu__upper95",

        "ell_mu__median",
        "ell_mu__lower68",
        "ell_mu__upper68",
        "ell_mu__lower95",
        "ell_mu__upper95",

        "A_sigma__median",
        "A_sigma__lower68",
        "A_sigma__upper68",
        "A_sigma__lower95",
        "A_sigma__upper95",

        "ell_sigma__median",
        "ell_sigma__lower68",
        "ell_sigma__upper68",
        "ell_sigma__lower95",
        "ell_sigma__upper95",

        "m_g__median",
        "m_g__lower68",
        "m_g__upper68",
        "m_g__lower95",
        "m_g__upper95",

        "diag__mean_inactive_probability",
        "diag__volatility_inactive_probability",

        "pred__expected_log_return",
        "pred__probability_up",
        "pred__terminal_sigma",

        "realized__terminal_log_return",
        "realized__horizon_realized_volatility",
    ]

    # Only use columns that actually exist.
    desired = [c for c in desired if c in df.columns]

    gp_mask = (
        df["model"]
        .astype(str)
        .str.contains("gp", case=False, na=False)
    )

    gp = df.loc[gp_mask, desired].copy()

    gp.to_csv(args.gp_output, index=False)

    print()
    print("=" * 80)
    print("GP PARAMETER HISTORY")
    print("=" * 80)
    print(f"records: {len(gp)}")
    print(f"saved:   {args.gp_output}")

    # ------------------------------------------------------------
    # Compact current snapshot to stdout
    # ------------------------------------------------------------

    latest = (
        df.sort_values("origin")
        .groupby(["model", "lookback"], as_index=False)
        .tail(1)
        .sort_values(["model", "lookback"])
    )

    snapshot_columns = [
        "origin",
        "model",
        "lookback",
        "status",
        "diag__converged",

        "diag__mu",
        "diag__sigma",

        "A_mu__median",
        "ell_mu__median",
        "A_sigma__median",
        "ell_sigma__median",
        "m_g__median",

        "max_rhat",
        "min_ess",
    ]

    snapshot_columns = [
        c for c in snapshot_columns
        if c in latest.columns
    ]

    print()
    print("=" * 80)
    print("LATEST PARAMETER SNAPSHOT")
    print("=" * 80)

    print(
        latest[snapshot_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
