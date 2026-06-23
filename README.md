# NumPy RL Racer

A from-scratch Deep Reinforcement Learning project where a small autonomous car learns
to navigate a 2D environment with obstacles toward a goal position.

The goal is to implement the environment, neural networks, and RL algorithms using NumPy only.

## Project constraints

Runtime dependencies:
- numpy
- matplotlib

Forbidden ML/RL dependencies:
- torch
- tensorflow
- jax
- gymnasium
- stable-baselines3

## Roadmap

### ✅ Done

1. **Project skeleton** — Python package structure (`pyproject.toml`, `src/`, `tests/`).
2. **2D car physics** — `KinematicCar` / `CarState`: position, heading, velocity,
   steering, acceleration, speed limits, heading normalisation.
3. **Track representation** — `RectangularTrack` with a configurable road width and
   point-to-segment distance checking.
4. **Racing environment API** — `RacingEnv` with `reset(seed)` / `step(action)` returning
   `(obs, reward, done, info)`; episode ends when the car leaves the track.
5. **Visualization** — `MatplotlibRenderer`: top‑down plot of the track with the car
   position and heading arrow, step counter and reward overlay.

### ✅ Done

6. **NumPy neural network** — Implement a feed‑forward network (Dense layers, ReLU,
   optional output activation) using only `numpy`, supporting forward‑pass,
   backward‑pass, and SGD optimisation.

7. **DQN from scratch** — Implement the Deep Q‑Network algorithm:
   - Experience replay buffer.
   - Epsilon‑greedy action selection.
   - Target network with periodic hard copy.
   - Q‑learning loss (MSE between target and online Q‑values).
   - Training loop (gradient descent via NumPy-only backpropagation or a simple
     optimiser like SGD).

### 📋 À faire (reste à faire)
8. **Training and evaluation scripts** — One or more scripts that:
   - Train the DQN agent in the `RacingEnv`.
   - Log episode rewards and losses.
   - Evaluate a trained policy (render a few rollouts).
   - Save / reload learned parameters.

## Quickstart

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint with ruff
uv run ruff check .
```
