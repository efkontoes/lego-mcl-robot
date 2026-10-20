# Legacy exploratory study (v1): historical record

This directory holds the **first, exploratory version** of the noise study, exactly as
it was produced at commit `df03434`. It is kept for transparency and is **not** the
evidence base for the current conclusions; see [`../README.md`](../README.md) and
[`docs/noise_study.md`](../../../docs/noise_study.md) for the revised study.

| File | What it is |
|---|---|
| `run_benchmark.py` | The v1 benchmark (closed-loop only, 30 seeds 0–29, variants V0-bug…V5) |
| `make_figures.py` | The v1 figure generator |
| `results/runs.csv`, `results/summary.md` | v1 per-run results and summary table |
| `results/steps.csv` | v1 per-update log (5 MB, not committed; regenerate with the script) |
| `results/figures/` | v1 figures |
| `noise_study_v1.md` | The v1 report, unchanged except for relative link paths |

## Why v1 is superseded

These are the protocol limitations that the revised study addresses:

1. **Closed loop only.** The localisation estimate drove the robot, so every variant
   saw a different trajectory, different motion noise and different sonar readings.
   Accuracy differences mixed estimator quality with navigation behaviour.
2. **Incomplete ablation.** Only V2, V3 and V4 (one technique each) and V5 (all three)
   were run. The pairwise combinations were missing, so the injection × KLD
   "interaction" was not actually tested.
3. **No particle-budget control.** V5's recovery advantage over V3 coincided with about
   8× more particles (785 vs 100).
4. **KLD implementation.** v1 drew `n_max` candidates by systematic resampling,
   shuffled them and truncated at the KLD bound. That does work proportional to
   `n_max` on every update and differs from Fox's sequential i.i.d. procedure. The
   revised code uses incremental KLD-sampling, so **v1 KLD timings are not comparable
   with v2 timings**.
5. **Seeds.** The 30 seeds (0–29) and the 12-seed injection-threshold tuning overlap
   with the development runs, so these results are exploratory, not held-out.
6. **Recovery metric.** Success was "final error < 10 cm" only. There was no sustained
   recovery and no censoring of runs that never recovered.

Re-running `run_benchmark.py` today would use the *corrected* KLD implementation and
therefore would not reproduce the v1 KLD numbers exactly. The CSVs here are the
authoritative record of v1.
