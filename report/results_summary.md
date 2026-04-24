# Results Summary

- Fairness controls: same dataset seed, same initialization seed, fixed epoch budgets, per-workload target-loss from a serial reference run, and identical logging cadence.
- Sparse, low-contention winner (runtime): serial (runtime_s=0.0354).
- Dense-update winner (runtime): serial (runtime_s=0.2413).
- High-contention winner (runtime): serial (runtime_s=0.0241).
- Hogwild vs striped-lock median runtime ratio at low rho (best 5 low-rho points): 1.095; high rho (top 5): 1.134.
- Model-size sweep completed; see `performance_vs_model_size_logistic.png` for scaling trend.
- Best time-to-target-loss in scaling runs: hogwild (time_to_target_s=0.1249).