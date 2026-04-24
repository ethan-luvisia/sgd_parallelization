#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def pick_best(df, metric, minimize=True):
    if df.empty:
        return None
    g = df.groupby("algorithm", as_index=False)[metric].median(numeric_only=True)
    g = g.sort_values(metric, ascending=minimize)
    return g.iloc[0]


def fmt_row(row, metric):
    return f"{row['algorithm']} ({metric}={row[metric]:.4f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--out", default="report/results_summary.md")
    args = p.parse_args()

    s = pd.read_csv(args.summary)

    lines = ["# Results Summary", ""]
    lines.append("- Fairness controls: same dataset seed, same initialization seed, fixed epoch budgets, per-workload target-loss from a serial reference run, and identical logging cadence.")

    sparse = s[s["run_id"].str.contains("logistic_sparsity_k4")]
    dense = s[s["run_id"].str.contains("logistic_sparsity_k200")]
    cont = s[s["run_id"].str.contains("logistic_contention_s1.6")]

    b1 = pick_best(sparse, "runtime_s", minimize=True)
    b2 = pick_best(dense, "runtime_s", minimize=True)
    b3 = pick_best(cont, "runtime_s", minimize=True)

    if b1 is not None:
        lines.append(f"- Sparse, low-contention winner (runtime): {fmt_row(b1, 'runtime_s')}.")
    if b2 is not None:
        lines.append(f"- Dense-update winner (runtime): {fmt_row(b2, 'runtime_s')}.")
    if b3 is not None:
        lines.append(f"- High-contention winner (runtime): {fmt_row(b3, 'runtime_s')}.")

    # relation between rho/delta and hogwild benefit
    hw = s[s["algorithm"] == "hogwild"]
    st = s[s["algorithm"] == "striped_lock"]
    merged = hw.merge(
        st,
        on=["trial", "workload", "threads", "active_k", "hotspot_skew", "dim", "mf_rank"],
        suffixes=("_hw", "_st"),
        how="inner",
    )
    if not merged.empty:
        merged["rt_ratio_hw_over_st"] = merged["runtime_s_hw"] / merged["runtime_s_st"]
        hi = merged.sort_values("rho_hw", ascending=False).head(5)
        lo = merged.sort_values("rho_hw", ascending=True).head(5)
        lines.append(
            f"- Hogwild vs striped-lock median runtime ratio at low rho (best 5 low-rho points): {lo['rt_ratio_hw_over_st'].median():.3f}; high rho (top 5): {hi['rt_ratio_hw_over_st'].median():.3f}."
        )

    # model size note
    ms = s[s["run_id"].str.contains("model_dim")]
    if not ms.empty:
        agg = ms.groupby(["algorithm", "dim"], as_index=False)["runtime_s"].median(numeric_only=True)
        lines.append("- Model-size sweep completed; see `performance_vs_model_size_logistic.png` for scaling trend.")

    # time-to-target
    ttt = s[(s["time_to_target_s"] > 0) & (s["run_id"].str.contains("scale"))]
    if not ttt.empty:
        best_ttt = pick_best(ttt, "time_to_target_s", minimize=True)
        lines.append(f"- Best time-to-target-loss in scaling runs: {fmt_row(best_ttt, 'time_to_target_s')}.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
