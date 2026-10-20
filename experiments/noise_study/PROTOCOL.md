# Noise study v2: pre-registered protocol

Written and frozen **before** any run on the evaluation seeds. Parameters come from
`config/robot.yaml` and the constants at the top of `study_lib.py`. Nothing is tuned on
seeds 1000–1099 or 2000–2029. The earlier 30-seed results (seeds 0–29) and the 12-seed
injection-threshold tuning are exploratory only (see `legacy_v1/`).

## Research question

Under what conditions do adaptive particle-filter techniques (ESS-gated resampling,
augmented-MCL injection, KLD-sampling) improve localisation accuracy and recovery, and
how do their interactions affect computation and closed-loop navigation reliability?

## Design

- **Factors:** ESS gating (E), injection (I), KLD-sampling (K), each on/off, on top of
  the odometry motion model (2³ = 8 cells, V1–V8).
- **Particle-budget controls:** V3-N500 and V3-N1000 (injection at a fixed 500/1000
  particles) and V1-N1000 (no technique, 1000 particles).
- **Historical controls:** V0 (coursework motion model). V0-bug is not re-run.

## Protocol A: fixed-trajectory replay (estimator claims come from here only)

- Reference traces come from a controller that uses the true pose. Every variant
  receives identical motions and sonar readings for a given seed.
- Seeds 1000–1099 (100 per scenario).
- **Scenarios:**
  - **Tracking:** known start, 8-waypoint course.
  - **Kidnap:** the course, plus a 30–50 cm and ±45° displacement after waypoint 3,
    then a 12-step sonar sweep at waypoint 3 and every later waypoint.
  - **Global:** unknown start, 1000 initial particles for every variant, 4 cycles of a
    sweep plus a short move.
- **Primary outcomes:**
  1. Kidnap: sustained recovery, meaning error < 10 cm for ≥ 5 consecutive updates
     after the kidnap, within the trace horizon. Unrecovered runs are right-censored.
  2. Tracking: mean position error over the run.
  3. Computation: total filter CPU time per trajectory (prediction + update).
- **Secondary outcomes:**
  - final error < 10 cm (historical definition)
  - updates to sustained recovery (Kaplan–Meier)
  - post-kidnap mean, median and 95th-percentile error
  - retained particle count
  - N_eff before resampling
  - occupied KLD bins
  - lineage diversity
  - injection probability
  - false confidence
  - credible-circle coverage
  - global convergence

## Protocol B: closed-loop navigation (navigation claims only)

- Seeds 1000–1099. Variants: V1, V2, V3, V4, V5, V6, V8, V3-N1000. Scenarios: tracking
  and kidnap.
- **Budgets:** 150 sonar updates (tracking) and 300 (kidnap); at most 40 attempts per
  waypoint.
- **Outcomes:**
  - waypoints declared reached
  - waypoints verified (true pose within 10 cm)
  - stalled waypoints
  - budget exhaustion
  - wall contacts
  - simulated time
  - updates and resamples
  - final error
  - post-kidnap sustained recovery
- Trajectories diverge between variants, so comparisons are not paired on sensor
  noise. Intervals are therefore unpaired (Wilson for proportions, a two-sample
  bootstrap for means).

## Noise-robustness grid (Protocol A)

- **Simulator conditions:** motion noise × {0.5, 1, 2} of nominal, crossed with sonar
  outlier probability {0, 0.03, 0.10}. Filter parameters stay at nominal.
- **Coverage:** variants V1, V2, V5 and V8; tracking and kidnap; seeds 2000–2029 per
  cell.

## Global-localisation diagnostic

The likelihood of the first 12-reading sweep is evaluated over a 2.5 cm × 5° grid of
start poses.

- **Plausible:** ll(truth) ≥ max ll − ln 100.
- **Ambiguous:** some cell more than 20 cm from the truth has max-over-heading
  ll ≥ ll(truth) − ln 10.
- **Failed global runs are classified as one of:**
  - **truth implausible:** the truth is not plausible.
  - **ambiguous:** the truth is plausible and the sweep is ambiguous.
  - **hypothesis lost:** the truth is plausible, the sweep is unambiguous, and the filter
    still failed.

## Prespecified analyses

1. **Factorial effects** on sustained kidnap recovery, tracking mean error and CPU per
   trajectory:
   - Main effects, two-way and three-way interactions, using ±1 contrasts over the 8
     cell means.
   - 95% CIs from a paired bootstrap over seeds (10,000 resamples).
2. **Simple effects:** injection with KLD off vs on (V3 − V1 against V8 − V4).
3. **Budget-matched comparisons:** V8 and V5 against V3-N1000 and V3-N500 on recovery,
   mean particle count and CPU per trajectory, as paired differences with bootstrap CIs.
4. **Proportions:** Wilson 95% intervals. Paired proportion differences use a seed
   bootstrap.
5. **Decision rule:** an effect is reported as *supported* only if its 95% CI excludes 0,
   and its practical size is stated in absolute units. An interaction is reported as
   *synergy* only if the I×K contrast CI excludes 0 **and** V8 beats the
   budget-matched control.

Everything else (stage-level CPU breakdown, false-confidence prevalence, the noise grid
heatmap, case-study trajectories) is **descriptive / exploratory**.

## Definitions

- **False confidence:** at an update, the 95% credible radius r95 ≤ 20 cm **and** the
  position error > 20 cm (the truth lies outside a tight credible circle).
- **Coverage:** the fraction of updates with error ≤ r95.
- **CPU per trajectory:** the sum over updates of predict + update wall time
  (`time.perf_counter`), measured on the machine recorded in `results/metadata/`.
  These are laptop timings, not Raspberry Pi timings.
