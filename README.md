# DiveSync Heavy

An autonomous underwater depth-control vehicle built around an ESP32 and MS5837 depth sensor, using a linear actuator syringe mechanism (0–100mm stroke) for buoyancy-based depth control. This repo contains the firmware, hardware design files, and control software — including PID and reinforcement learning (TD3) controllers — developed as part of an undergraduate research project under Umar.

## Hardware Overview

- **Depth sensing:** MS5837 pressure/depth sensor
- **Actuation:** Linear actuator-driven syringe mechanism, 0–100mm stroke, adjusts buoyancy by changing displaced volume
- **Compute:** ESP32, communicating over USB serial (115200 baud) with a host machine running the Python control stack
- **Control loop:** 20Hz telemetry/state/actuator loop, with an inner PID stabilizing actuator position and an outer controller (PID / BC / RL) setting the depth-tracking target

Full schematics, PCB layout, and BOM are under `/eda`.

## Repository Structure

```
divesync-heavy/
├── data/                     # Raw experiment logs, grouped by hardware/software version -- see data/README.md
├── eda/                      # KiCad schematic, PCB, BOM, and exported schematic PDF
├── firmware/                 # ESP32 firmware (PlatformIO project)
├── python/
│   ├── controllers/
│   │   ├── inner.py           # Inner PID loop — actuator position control
│   │   ├── manual.py          # Gamepad/keyboard manual control
│   │   ├── pid.py             # Outer PID depth controller
│   │   ├── rl.py              # TD3 (fully online) depth controller
│   │   └── waveform.py        # Square-wave setpoint/actuator generator
│   ├── core/
│   │   ├── experiment.py      # Experiment setup, handshake, state machine
│   │   ├── logger.py          # CSV logging (raw/processed/state)
│   │   ├── merger.py          # Merges state + processed CSVs into training_data.csv
│   │   ├── processor.py       # Depth filtering, actuator unit conversion
│   │   ├── serial_manager.py  # Serial I/O with the ESP32
│   │   ├── state.py           # Computes depth error, filtered velocity
│   │   └── telemetry.py       # Raw serial line parsing
│   ├── ml/
│   │   ├── reward.py              # TD3 reward function
│   │   ├── td3_env_spec.py        # Gym shape-declaration env for SB3 (also defines the fixed obs-normalization bounds)
│   │   ├── td3_model.zip          # Actively-training TD3 model (updated after each RL dive)
│   │   └── td3_replay_buffer.pkl  # Persisted replay buffer across dives
│   └── visualization/
│       ├── display.py         # Live Tkinter depth/setpoint/error display
│       └── plotter.py         # Post-dive matplotlib plotting (depth/actuator/motor voltage)
├── results/
│   ├── divesync-heavy-v1-results/   # Behavior cloning, cascaded PID, step response, sysid plots
│   ├── divesync-heavy-v2-results/   # Behavior cloning, cascaded PID (fixed/variable), RL (fixed/variable)
│   └── divesync-heavy-v3-results/   # Fully-online RL: early / failure / converged
├── main.py                    # Experiment entry point / main control loop
├── requirements.txt
├── .gitignore
└── README.md
```

## Control Strategies

Two outer-loop control strategies were implemented and compared:

1. **PID** — classical outer-loop depth PID feeding actuator position setpoints to the inner PID. Cascaded/nested PID across both loops was tested and found unstable (~99% of trials); the current architecture uses a single outer PID plus a fixed inner actuator-position PID instead.
2. **Reinforcement Learning (TD3)** — an off-policy, deterministic actor-critic algorithm (Twin Delayed DDPG), trained fully online on real hardware from a randomly-initialized actor and critics (no warm-start). Chosen over SAC/PPO/DDPG because its replay buffer suits the limited real-hardware sample budget, and its twin critics counter the Q-value overestimation that made vanilla DDPG unreliable.

### TD3 Implementation Notes

- Built on `stable_baselines3` (Gymnasium 1.3.0 / SB3 2.9.0), with a shape-only `DiveSyncEnvSpec` env used purely to satisfy SB3's constructor — no vectorized-env training loop is used. `main.py`'s own 20Hz loop *is* the environment loop; each dive is treated as one training episode.
- State inputs to the actor are normalized using the fixed bounds declared in `DiveSyncEnvSpec` (`OBS_LOW`/`OBS_HIGH`), not a data-fitted scaler.
- The reward function combines a depth-tracking error penalty, a velocity-damping penalty, and a smooth exponential barrier discouraging surfacing (and, in later iterations, an analogous barrier for the tank floor and an action-delta smoothness penalty).
- Both online (small periodic updates during a dive) and end-of-dive (`finalize()`, larger update) training were implemented; see **Known Issues** below regarding online training stability.

## Running an Experiment

```bash
pip install -r requirements.txt
python main.py
```

You'll be prompted for:
- **Control mode:** `manual` / `pid` / `rl` / `sysid`
- **Experiment duration** (seconds)
- **Notes** (optional, stored in `metadata.json`)
- **COM port** for the ESP32

Each run creates a timestamped folder under `data/` containing `raw.csv`, `processed.csv`, `state.csv`, `boot.txt`, and `metadata.json`. After the experiment ends, `training_data.csv` is generated (merging `state.csv` and `processed.csv`), and a results plot is shown automatically if `processed.csv` exists.

The first `rl`-mode run initializes a fresh TD3 model (random actor + critics) since there is no warm-start; `python/ml/td3_model.zip` and `td3_replay_buffer.pkl` are created after `finalize()` and reused/extended on subsequent RL dives.
