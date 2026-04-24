#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_case(exe, base, summary_csv, trace_csv, run_id, extra):
    cmd = [
        str(exe),
        "--run-id", run_id,
        "--out-summary-csv", str(summary_csv),
        "--out-trace-csv", str(trace_csv),
    ]
    for k, v in {**base, **extra}.items():
        cmd.extend([f"--{k}", str(v)])
    run_cmd(cmd)


def get_target_loss(exe, base, tmp_summary, tmp_trace, run_id, extra):
    if tmp_summary.exists():
        tmp_summary.unlink()
    if tmp_trace.exists():
        tmp_trace.unlink()

    run_case(exe, base, tmp_summary, tmp_trace, run_id, extra)

    with tmp_summary.open() as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["final_loss"])


def detect_threads(user_threads):
    if user_threads > 0:
        return sorted(set([t for t in [1, 2, 4, 8, user_threads] if t <= user_threads]))
    hw = os.cpu_count() or 4
    return sorted(set([t for t in [1, 2, 4, 8, hw] if t <= hw]))


def write_env(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"platform={platform.platform()}\n")
        f.write(f"python={sys.version.split()[0]}\n")
        f.write(f"cpu_count={os.cpu_count()}\n")
        try:
            cxx = subprocess.check_output(["c++", "--version"], text=True).splitlines()[0]
            f.write(f"compiler={cxx}\n")
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--exe", default="build/hogwild_bench")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--max-threads", type=int, default=0)
    args = p.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"benchmark executable not found: {exe}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "summary.csv"
    trace_csv = results_dir / "trace.csv"
    tmp_summary = results_dir / "_tmp_target_summary.csv"
    tmp_trace = results_dir / "_tmp_target_trace.csv"

    if summary_csv.exists():
        summary_csv.unlink()
    if trace_csv.exists():
        trace_csv.unlink()

    write_env(results_dir / "environment.txt")

    threads_list = detect_threads(args.max_threads)
    trials = [0, 1] if args.mode == "full" else [0]

    # Workload A: sparse logistic regression
    logistic_base = {
        "workload": "logistic",
        "epochs": 8 if args.mode == "full" else 5,
        "num-samples": 30000 if args.mode == "full" else 12000,
        "dim": 50000 if args.mode == "full" else 12000,
        "active-k": 10,
        "label-noise": 0.1,
        "hotspot-skew": 0.0,
        "l2": 1e-4,
        "schedule": "backoff",
        "lr": 0.05,
        "backoff-gamma": 0.9,
    }

    # Target from strong serial run with same budget
    target_loss = get_target_loss(
        exe,
        logistic_base,
        tmp_summary,
        tmp_trace,
        run_id="target_logistic_serial",
        extra={
            "algorithm": "serial",
            "threads": 1,
            "seed": 101,
            "init-seed": 202,
            "trial": 0,
        },
    )

    algorithms = ["serial", "coarse_lock", "striped_lock", "hogwild"]

    # Scaling + core sweeps
    for trial, algo, th in itertools.product(trials, algorithms, threads_list):
        if algo == "serial" and th != 1:
            continue
        run_case(
            exe,
            logistic_base,
            summary_csv,
            trace_csv,
            run_id=f"logistic_scale_t{th}_{algo}_trial{trial}",
            extra={
                "algorithm": algo,
                "threads": th,
                "seed": 1000 + trial,
                "init-seed": 2000 + trial,
                "trial": trial,
                "target-loss": target_loss,
            },
        )

    # Sparsity sweep
    for trial, k in itertools.product(trials, [4, 10, 40, 200]):
        for algo in ["serial", "striped_lock", "hogwild"]:
            th = threads_list[-1] if algo != "serial" else 1
            run_case(
                exe,
                logistic_base,
                summary_csv,
                trace_csv,
                run_id=f"logistic_sparsity_k{k}_{algo}_trial{trial}",
                extra={
                    "algorithm": algo,
                    "threads": th,
                    "active-k": k,
                    "seed": 3000 + trial,
                    "init-seed": 4000 + trial,
                    "trial": trial,
                    "target-loss": target_loss,
                },
            )

    # Contention sweep
    for trial, skew in itertools.product(trials, [0.0, 0.8, 1.2, 1.6]):
        for algo in ["serial", "striped_lock", "hogwild"]:
            th = threads_list[-1] if algo != "serial" else 1
            run_case(
                exe,
                logistic_base,
                summary_csv,
                trace_csv,
                run_id=f"logistic_contention_s{skew}_{algo}_trial{trial}",
                extra={
                    "algorithm": algo,
                    "threads": th,
                    "hotspot-skew": skew,
                    "seed": 5000 + trial,
                    "init-seed": 6000 + trial,
                    "trial": trial,
                    "target-loss": target_loss,
                },
            )

    # Model size sweep
    for trial, dim in itertools.product(trials, [2000, 12000, 50000]):
        for algo in ["serial", "striped_lock", "hogwild"]:
            th = threads_list[-1] if algo != "serial" else 1
            run_case(
                exe,
                logistic_base,
                summary_csv,
                trace_csv,
                run_id=f"logistic_model_dim{dim}_{algo}_trial{trial}",
                extra={
                    "algorithm": algo,
                    "threads": th,
                    "dim": dim,
                    "active-k": min(10, dim),
                    "seed": 7000 + trial,
                    "init-seed": 8000 + trial,
                    "trial": trial,
                    "target-loss": target_loss,
                },
            )

    # Schedule sweep
    for trial, schedule in itertools.product(trials, ["constant", "epoch_decay", "backoff"]):
        for algo in ["serial", "hogwild"]:
            th = threads_list[-1] if algo == "hogwild" else 1
            extra = {
                "algorithm": algo,
                "threads": th,
                "schedule": schedule,
                "seed": 9000 + trial,
                "init-seed": 10000 + trial,
                "trial": trial,
                "target-loss": target_loss,
            }
            if schedule == "epoch_decay":
                extra["decay"] = 0.4
            if schedule == "constant":
                extra["lr"] = 0.03
            run_case(
                exe,
                logistic_base,
                summary_csv,
                trace_csv,
                run_id=f"logistic_schedule_{schedule}_{algo}_trial{trial}",
                extra=extra,
            )

    # Workload B: matrix factorization
    mf_base = {
        "workload": "mf",
        "epochs": 6 if args.mode == "full" else 4,
        "mf-users": 4000 if args.mode == "full" else 1200,
        "mf-items": 5000 if args.mode == "full" else 1400,
        "mf-rank": 32 if args.mode == "full" else 16,
        "mf-observations": 350000 if args.mode == "full" else 70000,
        "mf-reg": 1e-4,
        "mf-noise": 0.05,
        "hotspot-skew": 0.8,
        "schedule": "backoff",
        "lr": 0.02,
        "backoff-gamma": 0.9,
    }

    mf_target = get_target_loss(
        exe,
        mf_base,
        tmp_summary,
        tmp_trace,
        run_id="target_mf_serial",
        extra={
            "algorithm": "serial",
            "threads": 1,
            "seed": 1101,
            "init-seed": 2202,
            "trial": 0,
        },
    )

    for trial, algo, th in itertools.product(trials, ["serial", "coarse_lock", "striped_lock", "hogwild"], threads_list):
        if algo == "serial" and th != 1:
            continue
        run_case(
            exe,
            mf_base,
            summary_csv,
            trace_csv,
            run_id=f"mf_scale_t{th}_{algo}_trial{trial}",
            extra={
                "algorithm": algo,
                "threads": th,
                "seed": 12000 + trial,
                "init-seed": 13000 + trial,
                "trial": trial,
                "target-loss": mf_target,
            },
        )

    # MF contention and model-size sweeps
    for trial, skew in itertools.product(trials, [0.0, 0.8, 1.4]):
        for algo in ["serial", "striped_lock", "hogwild"]:
            th = threads_list[-1] if algo != "serial" else 1
            run_case(
                exe,
                mf_base,
                summary_csv,
                trace_csv,
                run_id=f"mf_contention_s{skew}_{algo}_trial{trial}",
                extra={
                    "algorithm": algo,
                    "threads": th,
                    "hotspot-skew": skew,
                    "seed": 14000 + trial,
                    "init-seed": 15000 + trial,
                    "trial": trial,
                    "target-loss": mf_target,
                },
            )

    for trial, rank in itertools.product(trials, [8, 16, 32, 64]):
        for algo in ["serial", "striped_lock", "hogwild", "local_batch_reduce"]:
            th = threads_list[-1] if algo != "serial" else 1
            run_case(
                exe,
                mf_base,
                summary_csv,
                trace_csv,
                run_id=f"mf_rank_r{rank}_{algo}_trial{trial}",
                extra={
                    "algorithm": algo,
                    "threads": th,
                    "mf-rank": rank,
                    "seed": 16000 + trial,
                    "init-seed": 17000 + trial,
                    "trial": trial,
                    "target-loss": mf_target,
                },
            )

    if tmp_summary.exists():
        tmp_summary.unlink()
    if tmp_trace.exists():
        tmp_trace.unlink()

    print(f"Wrote results to {summary_csv} and {trace_csv}")


if __name__ == "__main__":
    main()
