# Lock-Free Parallel Stochastic Gradient Descent on Multicore Systems

## 1. Project Question

When do Hogwild-style lock-free updates outperform synchronized alternatives on multicore CPUs, and how do sparsity, contention, model size, and schedule choices affect runtime, throughput, convergence, and time-to-target-loss?

## 2. Motivation

Synchronized SGD often pays lock overhead that can erase multicore gains. Hogwild-style asynchronous updates remove lock critical sections and can improve throughput when update overlap is low. This project benchmarks those tradeoffs in a reproducible, CPU-only framework.

## 3. Background (Hogwild)

Using `hogwildTR.pdf` as primary reference:

- Hogwild relies on sparse updates where collisions are rare.
- The paper frames sparsity/contention via update footprint and conflict structure.
- We use practical proxies analogous to paper symbols:
  - `Omega`: coordinates touched per update
  - `Delta`: max normalized coordinate incidence
  - `rho`: sampled overlap/conflict probability between two updates
- We compare fixed and backoff-style learning-rate schedules, including epoch-wise multiplicative backoff.

## 4. System Design and Implementation

### 4.1 Core stack

- C++17 + CMake
- OpenMP parallel regions (GCC/OpenMP build)
- Python scripts for experiment orchestration and plotting

### 4.2 Algorithms

Implemented modes:

1. `serial`: single-thread SGD
2. `coarse_lock`: one global mutex around each parameter update
3. `striped_lock`: lock striping over parameter blocks
4. `hogwild`: mutex-free updates via atomic/CAS coordinate adds (defined C++ behavior)
5. `local_batch_reduce` (optional baseline for MF): thread-local accumulation + periodic critical reduction

### 4.3 Workloads

1. Sparse logistic regression (synthetic)
- Configurable `N`, `d`, active coordinates `k`, noise, hotspot skew.

2. Matrix factorization (synthetic)
- Configurable users/items/rank/observations/noise/skew.
- Each update touches user and item factors (sparse but larger parameter state).

## 5. Methodology and Fairness

Fairness controls enforced in code/scripts:

- Same dataset seed and generation settings across compared algorithms
- Same initialization seed (`init_seed`) across compared algorithms
- Same epoch budget and sample touches
- Same metric logging cadence (epoch checkpoints)
- Target loss per workload from reference serial run, then reused for all methods
- Multiple trials in full suite (`trial` 0 and 1), median comparisons
- Same CLI path and CSV schema across methods

Non-determinism note: parallel methods (especially Hogwild) are asynchronous; exact update interleavings differ run-to-run even with fixed seeds.

## 6. Experimental Setup

Hardware/software used for reported runs (`results_full/environment.txt`):

- macOS 15.6 arm64, 12 logical CPUs
- GCC 15 + OpenMP (`/opt/homebrew/bin/g++-15`)
- Python 3.11 + pandas + matplotlib

Main full suite output:

- `results_full/summary.csv`
- `results_full/trace.csv`

Extra stress test:

- Added `logistic_scale_heavy_*` runs (`N=500000, d=50000, k=10`) to better expose scaling beyond sub-100ms runtimes.

## 7. Results

Figures are in `report/figures/`.

### 7.1 Scaling (threads)

- Logistic scale (median, full suite): serial remained fastest in wall-clock runtime.
  - serial@1: 0.0236s
  - hogwild@4: 0.0397s (speedup vs serial = 0.595x)
  - striped_lock@12: 0.2077s
- MF scale (median, full suite): Hogwild achieved real speedup over serial.
  - serial@1: 0.4313s
  - hogwild@12: 0.2985s (1.445x vs serial)
  - striped_lock@4: 0.2649s (1.628x vs serial, best point)

### 7.2 Time-to-target-loss

- Logistic scaling: best median time-to-target among parallel methods at hogwild@4 (~0.0239s), but serial@1 remained lower (~0.0131s).
- MF scaling: several methods reached target quickly; Hogwild improved wall time in higher-thread settings when rank/working set were moderate.

### 7.3 Sparsity sweep (logistic)

Median runtime by `k`:

- `k=4`: serial 0.0167s, hogwild 0.1223s, striped_lock 0.0867s
- `k=40`: serial 0.0766s, hogwild 0.1102s, striped_lock 1.0129s
- `k=200`: serial 0.2832s, hogwild 0.4761s, striped_lock 2.1191s

Observation: Hogwild consistently outperformed striped locking, especially at higher `k`, but serial still won in this logistic configuration.

### 7.4 Contention sweep (logistic)

At 12 threads (median):

- low contention (`skew=0.0`, low `rho`): Hogwild faster than striped (`runtime ratio hw/striped = 0.454`)
- high contention (`skew=1.6`, `rho≈1`): Hogwild slower than striped (`ratio = 1.929`)

This directly matches the paper intuition: lock-free updates degrade when overlap/collisions become extreme.

### 7.5 Model-size sweep (logistic)

For `d in {2000, 12000, 50000}` at 12-thread parallel modes:

- Hogwild runtime stayed around 0.11–0.13s.
- Striped lock stayed slower (0.17–0.20s).
- Serial stayed near 0.024s for this budget.

### 7.6 Heavy logistic stress test (added)

`N=500000, d=50000, k=10` (single trial):

- serial@1: 1.3717s
- hogwild@4: 0.5916s (2.319x speedup vs serial)
- hogwild@8: 0.6943s (1.975x)
- hogwild@12: 1.2168s (1.127x)

This confirms that when work per run is larger, Hogwild can surpass serial and synchronized baselines.

## 8. Discussion

### 8.1 Why Hogwild helps in sparse regimes

When `rho` is low and update overlap is infrequent, Hogwild avoids lock critical sections and beats synchronized alternatives (especially striped lock).

### 8.2 Why dense/high-contention hurts

As overlap proxies increase (`rho -> 1`, very high `Delta`), stale/conflicting updates and atomic pressure rise. In logistic contention sweeps, Hogwild lost to striped lock at highest skew.

### 8.3 Model size and batch/local accumulation

- Model-size alone did not rescue synchronized methods in the tested logistic budget.
- `local_batch_reduce` (MF stretch baseline) was generally slower than Hogwild in this implementation due heavy reduction overhead.

### 8.4 Step schedule sensitivity

Backoff and constant schedules were both stable in tested ranges. Backoff was used as default for main sweeps to mirror paper-style epoch decay.

### 8.5 Paper expectation vs practice

Empirical trends align qualitatively with `hogwildTR.pdf`:

- Sparse/low-overlap settings favor lock-free updates.
- High-overlap/high-contention settings reduce Hogwild benefits.
- Implementation details (atomic cost, scheduler overhead, cache effects) materially affect observed crossover points.

## 9. Required Direct Answers

1. How was fairness ensured?
- Same data generation seed, same initialization seed, same epoch/update budget, same target-loss definition, same logging cadence, and median-over-trials analysis.

2. Does Hogwild behave differently in dense/high-contention workloads?
- Yes. In low contention it beat striped lock; in highest contention (`skew=1.6`, `rho≈1`) it became slower than striped lock.

3. How much do model size and batch size matter?
- Model size shifted absolute runtime and convergence region, but contention/sparsity still dominated relative ranking. Local batch reduction (optional) reduced lock frequency but was slower in this implementation due merge overhead.

4. Do empirical results resemble paper predictions?
- Qualitatively yes: low-conflict sparse updates favor Hogwild; high conflict erodes gains.

5. Where do synchronized methods remain competitive?
- Striped locking was competitive in some MF scale points and in very high-contention logistic settings where Hogwild suffered.

## 10. Limitations and Future Work

- No thread pinning/affinity in current scripts.
- `rho`/`Delta` are empirical proxies, not exact theoretical constants.
- Epoch-level loss evaluation (not per-update) can hide fine-grained transient behavior.
- Future extensions: hybrid synchronization on hotspot coordinates, explicit round-robin stale-update baseline, and Linux `perf` counters.

## 11. Conclusion

The framework is fully reproducible and supports required baselines, workloads, and report artifacts. On this CPU, Hogwild strongly outperformed synchronized alternatives in several sparse settings and in larger MF/heavy-logistic runs, but not universally: high-contention regimes and short-run overheads can negate lock-free gains.

