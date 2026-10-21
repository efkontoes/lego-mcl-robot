# Noise study: experiment code and data

The report is in [`docs/noise_study.md`](../../docs/noise_study.md); this file covers the
mechanics.

## Reproduce

From the repository root, with the dev extras installed (`pip install -e ".[dev]"`):

```bash
cd experiments/noise_study
python run_study.py all     # all simulation batches, ~15 min on a laptop; resumable
python analyze.py           # statistics -> results/analysis/ (no simulation)
python make_figures.py      # figures -> results/figures/ (no simulation)
```

Batches can also be run individually: `traces`, `main`, `closed_loop`, `noise`,
`global_diag`.
- Each batch writes atomic part files (one per 10-seed block) and skips parts that
  already exist, so an interrupted run resumes where it stopped.
- Delete `results/` to start from scratch.

## Layout

| Path | Contents |
|---|---|
| `PROTOCOL.md` | Design, outcomes and decision rules, frozen before the evaluation runs |
| `study_lib.py` | Trace generation (true-pose controller), replay (Protocol A), closed-loop runner (Protocol B), metrics, Wilson/bootstrap/Kaplan–Meier, metadata |
| `run_study.py` | Batch runner |
| `analyze.py` | Per-run metrics, intervals, factorial contrasts, budget comparisons → `results/analysis/` |
| `make_figures.py` | The 11 figures, reading only saved results |
| `animations/` | ManimGX animations built from the saved results (see its README) |
| `results/traces/` | Reference traces (gzipped JSON): every motion, sonar reading, true pose and kidnap event |
| `results/raw/main/` | Protocol A per-update logs (seeds 1000–1099), particle snapshots for seeds 1000–1009 |
| `results/raw/closed_loop/` | Protocol B per-run summaries and per-update logs |
| `results/raw/noise/` | Noise-grid per-update logs (seeds 2000–2029) |
| `results/raw/global_diag/` | First-sweep likelihood grid summaries and maps |
| `results/analysis/` | Derived tables; `summary.md` is the human-readable digest |
| `results/metadata/` | Per-batch seeds, configuration (with SHA-256), git revision, environment, command |
| `results/figures/` | Figures used in the report |
| `results/run_log.txt` | Console log of the one complete run |
| `legacy_v1/` | The original exploratory study (v1), kept unchanged for transparency |

## Seeds

| Seeds | Use |
|---|---|
| 0–29 | Development and v1 (exploratory; the 12-seed injection-threshold tuning was here) |
| 1000–1099 | Held-out main evaluation (Protocols A and B, global diagnostic) |
| 2000–2029 | Noise-robustness grid |
