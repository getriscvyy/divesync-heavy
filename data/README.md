# Data Organization

Raw experiment logs, grouped to match the hardware/software progression documented in `/results`. Each leaf folder is one experiment run (`YYYYMMDD-HHMMSS`), containing `raw.csv`, `processed.csv`, `state.csv`, `boot.txt`, `metadata.json`, and (once merged) `training_data.csv`.

## `divesync-heavy-v1-data/` (2026-07-06 -- 2026-07-09, 14 dives)

Earliest hardware iteration (50mm actuator stroke). Manual piloting, initial PID characterization, step-response/system-ID runs, and the first behavior-cloning + BC-warmstarted-RL attempts. Corresponds to `results/divesync-heavy-v1-results/`.

## `divesync-heavy-v2-data/` (2026-07-10, 2026-07-14, 31 dives)

100mm actuator stroke. Manual piloting for BC training data, PID validation (fixed/variable setpoint), and BC-pretraining-warmstarted TD3 -- the warmstart approach later found to underperform and replaced in v3. Corresponds to `results/divesync-heavy-v2-results/`.

## `divesync-heavy-v3-data/` -- fully-online TD3, no BC warm-start

The BC/warmstart pipeline was removed; TD3 trains entirely online from real dives from here on. Corresponds to `results/divesync-heavy-v3-results/` (reinforcement-learning-early/failure/converged). Split into three subfolders:

- **`pid-validation/`** (2026-07-16, 11 dives) -- pure PID control-mode dives, re-validating the classical baseline on this hardware revision before RL work resumed.
- **`pre-swap/`** (2026-07-17, 23 dives) -- first fully-online TD3 dives on this hardware. Includes the full progression from cold-start collapse through the stuck-recovery and velocity-penalty fixes; best sustained result is `20260717-161502` (81% of samples within 0.05m of setpoint over a 10-minute dive).
- **`post-swap/`** (2026-07-21 -- 2026-07-22, 15 dives) -- after the actuator/buoyancy device was physically swapped, changing the depth dynamics enough to require a model reset (the pre-swap model/replay buffer no longer matched reality -- see `divesync-heavy-v3-data-incomplete-runs/README` for how that was diagnosed). Training restarted fully online on this dataset, converging faster than the original cold start since the warmup/stuck-recovery/reward fixes were already in place.

**Note for reuse:** `pre-swap` and `post-swap` reflect two physically different dynamical systems (same v3 software/methodology, different hardware buoyancy characteristics). Don't pool them for a single tracking-accuracy statistic without accounting for that split.
