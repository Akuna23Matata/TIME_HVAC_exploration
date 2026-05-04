# TIME: Information-Theoretic MPPI Exploration for Efficient HVAC Control

**TIME** (Information-Theoretic MPPI Exploration, reversed) is an exploration-enhanced Model Predictive Path Integral (MPPI) controller for HVAC systems. It actively reduces dynamics-model uncertainty by choosing setpoint actions that maximize information gain over a curated set of "exploration target" state-action points (the **Z-dataset**), in addition to optimizing comfort and energy.

The system is built on top of [Sinergym](https://github.com/ugr-sail/sinergym) (a `gymnasium` wrapper around EnergyPlus) and uses a Gaussian Process model of the indoor temperature dynamics for both prediction and uncertainty quantification.

## System Architecture

Three core algorithmic components:

1. **Z-Dataset** ([`exploration_mppi/zdataset.py`](exploration_mppi/zdataset.py)) — maintains exploration target points `z = [state, action] ∈ ℝ^10`.
   - **Type 1** (cold start): biased toward comfort-boundary temperatures.
   - **Type 2** (operational): added when the controller encounters high-uncertainty states during deployment.
2. **Gaussian Process model** ([`exploration_mppi/gp.py`](exploration_mppi/gp.py)) — predicts indoor-temperature delta `Δt` from `(state, action)` with `RBF + WhiteKernel`, and exposes posterior variance for uncertainty.
3. **Exploration MPPI controller** ([`exploration_mppi/controller.py`](exploration_mppi/controller.py)) — extends the standard MPPI baseline ([`exploration_mppi/mppi_baseline.py`](exploration_mppi/mppi_baseline.py)) with an information-gain term in the trajectory score and an `exploration_flag ∈ [0, 1]` knob mixing exploration vs. exploitation. Uses an analytical approximation for mutual information.

### Modified reward

```
r = f_e * (information_gain - 0.01 * E_t) + (1 - f_e) * (w_e * (-E_t) - (1 - w_e) * comfort_violation)
```

- `f_e = 0` → pure exploitation (standard MPPI).
- `f_e = 1` → pure exploration (no uncertainty penalty).
- `f_e ∈ (0, 1)` → balanced.

### Information gain (approximation)

For a single exploration target `z`, the mutual information of observing `y` at `x` is
```
I({x, y}; Ψ(z)) = H[Ψ(z)] - E_y[H[Ψ(z) | {x, y}]]
```
with closed-form prior and expected-posterior entropies (see [`exploration_mppi/gp.py`](exploration_mppi/gp.py)). For a set of targets `Z = {z_1, …, z_m}`, the total information gain is a (weighted) sum across targets.

## Experiments

Two experiment protocols ship in this repo.

### A — Three-week protocol (`experiments/three_week/`)

| File | Role |
|------|------|
| `run_exploration.py` | **TIME** — exploration MPPI runner with daily GP retraining during week 2. |
| `run_clue_baseline.py` | Rule-based + standard MPPI ("CLUE") baseline. |
| `run_random_baseline.py` | Random-action baseline. |

Protocol:
1. **Week 1** — rule-based controller bootstraps the GP (~672 samples).
2. **Week 2/3** — exploration MPPI explores; GP retrains daily as new data arrives.
3. **Week 4** — standard MPPI evaluates on the retrained GP.

```bash
python experiments/three_week/run_exploration.py --summer
```

### B — Ten-week protocol (`experiments/ten_week/`)

| File | Role |
|------|------|
| `run_exploration.py` | Ten-week runner: weekdays run standard MPPI control, weekends do information-gain exploration. |
| `run_clue_baseline.py` | Ten-week CLUE baseline. |
| `run_random_baseline.py` | Ten-week random baseline. |

Constants: `TOTAL_EXPERIMENT_WEEKS = 10`, `WEEKDAY_CONTROL_DAYS = 5`, `WEEKEND_EXPLORATION_DAYS = 2`.

```bash
python experiments/ten_week/run_exploration.py --summer
```

## Installation

The non-trivial part of installation is EnergyPlus + Sinergym. Follow [Sinergym's install guide](https://ugr-sail.github.io/sinergym/source/installation.html) first.

## Repo Layout

```
TIME_HVAC_exploration/
├── exploration_mppi/         # Core library
│   ├── controller.py         # Exploration MPPI
│   ├── mppi_baseline.py      # Standard MPPI baseline
│   ├── gp.py                 # GP dynamics + information gain
│   ├── zdataset.py           # Z-dataset management
│   └── args.py               # CLI argument parsing
├── experiments/
│   ├── three_week/           # 3-week protocol scripts
│   └── ten_week/             # 10-week protocol scripts
```

## License

MIT — see [LICENSE](LICENSE).
