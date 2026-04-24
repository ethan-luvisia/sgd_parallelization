#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def median_group(df, by, cols):
    return df.groupby(by, as_index=False)[cols].median(numeric_only=True)


def plot_speedup(summary, out_dir, workload, pattern):
    sub = summary[(summary["workload"] == workload) & summary["run_id"].str.contains(pattern)]
    if sub.empty:
        return
    serial = sub[(sub["algorithm"] == "serial") & (sub["threads"] == 1)]
    if serial.empty:
        return
    serial_rt = serial["runtime_s"].median()
    g = median_group(sub[sub["algorithm"] != "serial"], ["algorithm", "threads"], ["runtime_s"])
    g["speedup"] = serial_rt / g["runtime_s"]

    plt.figure(figsize=(7, 4))
    for algo, df_algo in g.groupby("algorithm"):
        plt.plot(df_algo["threads"], df_algo["speedup"], marker="o", label=algo)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Threads")
    plt.ylabel("Speedup vs serial")
    plt.title(f"Speedup vs Threads ({workload})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"speedup_threads_{workload}.png", dpi=160)
    plt.close()


def plot_throughput(summary, out_dir, workload, pattern):
    sub = summary[(summary["workload"] == workload) & summary["run_id"].str.contains(pattern)]
    if sub.empty:
        return
    g = median_group(sub, ["algorithm", "threads"], ["throughput_updates_per_s"])
    plt.figure(figsize=(7, 4))
    for algo, df_algo in g.groupby("algorithm"):
        plt.plot(df_algo["threads"], df_algo["throughput_updates_per_s"], marker="o", label=algo)
    plt.xlabel("Threads")
    plt.ylabel("Throughput (updates/s)")
    plt.title(f"Throughput vs Threads ({workload})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"throughput_threads_{workload}.png", dpi=160)
    plt.close()


def plot_loss_curves(trace, out_dir, workload, pattern, x_col, fname, xlabel):
    sub = trace[(trace["workload"] == workload) & trace["run_id"].str.contains(pattern)]
    if sub.empty:
        return

    # representative trial: median trial id if available
    if "trial" in sub.columns:
        try:
            trial = int(sub["trial"].median())
            sub = sub[sub["trial"] == trial]
        except Exception:
            pass

    plt.figure(figsize=(7, 4))
    for algo, df_algo in sub.groupby("algorithm"):
        med = df_algo.groupby(x_col, as_index=False)["loss"].median(numeric_only=True)
        plt.plot(med[x_col], med["loss"], marker="o", label=algo)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(f"Loss Curves ({workload})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=160)
    plt.close()


def plot_ttt(summary, out_dir, workload, pattern):
    sub = summary[(summary["workload"] == workload) & summary["run_id"].str.contains(pattern)]
    sub = sub[sub["algorithm"] != "serial"]
    if sub.empty:
        return
    g = median_group(sub, ["algorithm", "threads"], ["time_to_target_s"])
    plt.figure(figsize=(7, 4))
    for algo, df_algo in g.groupby("algorithm"):
        plt.plot(df_algo["threads"], df_algo["time_to_target_s"], marker="o", label=algo)
    plt.xlabel("Threads")
    plt.ylabel("Time to target loss (s)")
    plt.title(f"Time-to-Target vs Threads ({workload})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"ttt_threads_{workload}.png", dpi=160)
    plt.close()


def plot_sweep(summary, out_dir, workload, key, pattern, fname, y_col, ylabel, title):
    sub = summary[(summary["workload"] == workload) & summary["run_id"].str.contains(pattern)]
    if sub.empty:
        return
    g = median_group(sub, ["algorithm", key], [y_col])
    plt.figure(figsize=(7, 4))
    for algo, df_algo in g.groupby("algorithm"):
        plt.plot(df_algo[key], df_algo[y_col], marker="o", label=algo)
    plt.xlabel(key)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=160)
    plt.close()


def write_summary_table(summary, out_path):
    lines = ["# Results Summary Table", ""]

    def winner(df, metric, minimize=False):
        if df.empty:
            return "n/a"
        g = df.groupby("algorithm", as_index=False)[metric].median(numeric_only=True)
        row = g.sort_values(metric, ascending=minimize).iloc[0]
        return f"{row['algorithm']} ({metric}={row[metric]:.4f})"

    sparse = summary[summary["run_id"].str.contains("logistic_sparsity_k4")]
    dense = summary[summary["run_id"].str.contains("logistic_sparsity_k200")]
    high_cont = summary[summary["run_id"].str.contains("logistic_contention_s1.6")]

    lines.append("| Question | Winner |")
    lines.append("|---|---|")
    lines.append(f"| Fastest sparse low-contention setting | {winner(sparse, 'runtime_s', minimize=True)} |")
    lines.append(f"| Fastest dense/high-contention setting | {winner(dense, 'runtime_s', minimize=True)} |")
    lines.append(f"| Best throughput under high contention | {winner(high_cont, 'throughput_updates_per_s', minimize=False)} |")
    lines.append(f"| Best time-to-target-loss (scaling runs) | {winner(summary[summary['run_id'].str.contains('scale')], 'time_to_target_s', minimize=True)} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--trace", default="results/trace.csv")
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--report-fig-dir", default="report/figures")
    args = p.parse_args()

    summary = pd.read_csv(args.summary)
    trace = pd.read_csv(args.trace)

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.report_fig_dir)
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    plot_speedup(summary, out_dir, "logistic", "logistic_scale")
    plot_speedup(summary, out_dir, "matrix_factorization", "mf_scale")
    plot_throughput(summary, out_dir, "logistic", "logistic_scale")
    plot_throughput(summary, out_dir, "matrix_factorization", "mf_scale")
    plot_ttt(summary, out_dir, "logistic", "logistic_scale")
    plot_ttt(summary, out_dir, "matrix_factorization", "mf_scale")

    plot_loss_curves(trace, out_dir, "logistic", "logistic_scale", "elapsed_s", "loss_vs_time_logistic.png", "Wall-clock time (s)")
    plot_loss_curves(trace, out_dir, "logistic", "logistic_scale", "updates", "loss_vs_updates_logistic.png", "Processed updates")
    plot_loss_curves(trace, out_dir, "matrix_factorization", "mf_scale", "elapsed_s", "loss_vs_time_mf.png", "Wall-clock time (s)")

    plot_sweep(
        summary,
        out_dir,
        "logistic",
        "active_k",
        "logistic_sparsity",
        "performance_vs_sparsity_logistic.png",
        "runtime_s",
        "Runtime (s)",
        "Performance vs Update Sparsity (logistic)",
    )
    plot_sweep(
        summary,
        out_dir,
        "logistic",
        "hotspot_skew",
        "logistic_contention",
        "performance_vs_contention_logistic.png",
        "runtime_s",
        "Runtime (s)",
        "Performance vs Contention (logistic)",
    )
    plot_sweep(
        summary,
        out_dir,
        "logistic",
        "dim",
        "logistic_model_dim",
        "performance_vs_model_size_logistic.png",
        "runtime_s",
        "Runtime (s)",
        "Performance vs Model Size (logistic)",
    )

    for png in out_dir.glob("*.png"):
        target = fig_dir / png.name
        target.write_bytes(png.read_bytes())

    write_summary_table(summary, Path("results/summary_table.md"))


if __name__ == "__main__":
    main()
