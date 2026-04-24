# Lock-Free Parallel SGD on Multicore Systems (CSC 2/458 Spring 2026)

This repository contains a reproducible CPU-only benchmarking framework comparing:

1. `serial` SGD
2. `coarse_lock` synchronized parallel SGD
3. `striped_lock` synchronized parallel SGD with lock striping
4. `hogwild` mutex-free asynchronous SGD (implemented with atomic/CAS coordinate updates; no undefined data races)
5. optional `local_batch_reduce` baseline (matrix factorization workload)

The implementation follows the `hogwildTR.pdf` reference in this repo for sparsity/contention framing, backoff schedule, and fair comparison design.

## Repository Layout

- `src/`, `include/`: C++17 training core and CLI benchmark
- `tests/`: smoke/correctness tests + CSV schema regression test
- `scripts/`: experiment runner + plotting + report-summary generation
- `results/`, `results_full/`: generated benchmark CSV outputs
- `plots/`: generated figures
- `report/`: report draft and figure copies

## Dependencies

- CMake >= 3.16
- C++17 compiler
- OpenMP-capable compiler for parallel runs (recommended: Homebrew GCC on macOS)
- Python 3.10+
- Python packages: `pandas`, `matplotlib`

## Build

Recommended (OpenMP-enabled on macOS):

```bash
cmake -S . -B build_gcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15
cmake --build build_gcc -j
```

Fallback:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run Tests

```bash
ctest --test-dir build_gcc --output-on-failure
```

## Benchmark CLI Examples

Sparse logistic, Hogwild 8 threads:

```bash
build_gcc/hogwild_bench \
  --workload logistic --algorithm hogwild --threads 8 \
  --epochs 8 --num-samples 30000 --dim 50000 --active-k 10 \
  --schedule backoff --lr 0.05 --backoff-gamma 0.9 \
  --out-summary-csv results/summary.csv --out-trace-csv results/trace.csv
```

Matrix factorization, striped lock 12 threads:

```bash
build_gcc/hogwild_bench \
  --workload mf --algorithm striped_lock --threads 12 \
  --epochs 6 --mf-users 4000 --mf-items 5000 --mf-rank 16 --mf-observations 350000 \
  --schedule backoff --lr 0.02 --backoff-gamma 0.9 \
  --out-summary-csv results/summary.csv --out-trace-csv results/trace.csv
```

## Reproducibility Commands

Quick smoke benchmark suite:

```bash
python3 scripts/run_benchmarks.py --mode quick --exe build_gcc/hogwild_bench --results-dir results
```

Full benchmark suite (used for report figures):

```bash
python3 scripts/run_benchmarks.py --mode full --exe build_gcc/hogwild_bench --results-dir results_full
```

Generate plots:

```bash
HOME=/tmp MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl_cache XDG_CACHE_HOME=/tmp \
python3 scripts/plot_results.py \
  --summary results_full/summary.csv \
  --trace results_full/trace.csv \
  --out-dir plots \
  --report-fig-dir report/figures
```

Generate report summary bullets:

```bash
python3 scripts/generate_report_summary.py --summary results_full/summary.csv --out report/results_summary.md
```

## Make Targets

- `make build`
- `make test`
- `make quick`
- `make full`
- `make plots`
- `make report`

## Fairness Controls Implemented

- Same synthetic dataset instance per run configuration
- Same random initialization seed per compared algorithms
- Same epoch/sample-touch budget
- Same evaluation cadence (epoch checkpoints)
- Same target-loss definition per workload from reference serial runs
- Trials (`trial` column) and median-based analysis
- Explicit logging of `omega`, `delta`, `rho` contention/sparsity proxies

## Environment Used for Reported Runs

From `results_full/environment.txt`:

- Timestamp: 2026-04-24 19:20:26 EDT
- Platform: macOS 15.6 arm64
- CPU threads: 12
- Compiler toolchain for reported runs: GCC 15 (`/opt/homebrew/bin/g++-15`, OpenMP enabled)

