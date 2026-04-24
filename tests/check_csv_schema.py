#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_csv_schema.py <bench_exe>")
    exe = Path(sys.argv[1])
    out_dir = Path("tests/_tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.csv"
    trace = out_dir / "trace.csv"
    if summary.exists():
        summary.unlink()
    if trace.exists():
        trace.unlink()

    cmd = [
        str(exe),
        "--workload", "logistic",
        "--algorithm", "serial",
        "--threads", "1",
        "--epochs", "2",
        "--num-samples", "200",
        "--dim", "100",
        "--active-k", "4",
        "--run-id", "schema_test",
        "--out-summary-csv", str(summary),
        "--out-trace-csv", str(trace),
    ]
    subprocess.run(cmd, check=True)

    required_summary = {
        "run_id", "trial", "seed", "init_seed", "workload", "algorithm", "schedule", "threads", "epochs",
        "runtime_s", "total_updates", "throughput_updates_per_s", "final_loss", "omega", "delta", "rho"
    }
    required_trace = {"run_id", "workload", "algorithm", "threads", "epoch", "elapsed_s", "updates", "loss"}

    with summary.open() as f:
        header = set(next(csv.reader(f)))
        missing = required_summary - header
        if missing:
            raise SystemExit(f"missing summary columns: {sorted(missing)}")

    with trace.open() as f:
        header = set(next(csv.reader(f)))
        missing = required_trace - header
        if missing:
            raise SystemExit(f"missing trace columns: {sorted(missing)}")


if __name__ == "__main__":
    main()
