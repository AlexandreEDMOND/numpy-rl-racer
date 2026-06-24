# NumPy RL Racer

A from-scratch Deep Reinforcement Learning project where a small autonomous car learns to navigate 2D tracks using only NumPy. No PyTorch, TensorFlow, JAX, Gymnasium, or any external RL library.

## What this repo is about

This project implements every component of a deep RL system — kinematic car physics, track geometries, neural networks, optimisers, and DQN-based agents — using **NumPy only**. It serves as:

- A **learning resource** for understanding how deep RL works under the hood, from gradient descent through Q-learning to modern improvements like Double DQN, Dueling networks, Prioritised Experience Replay, Noisy Networks, and N-step returns.
- An **educational benchmark** for experimenting with RL algorithms in a simple, self-contained 2D racing environment.

## Features

**Environment:**
- Kinematic bicycle-model car with steering, acceleration, speed limits
- Multiple track types: rectangular, circular, figure-8
- Configurable track width, obstacles, randomised start positions
- Lap-based reward with progress tracking and reward lines (checkpoint gates)
- Observation includes position, heading, velocity, distance-to-edge, heading error, and optional obstacle sensing

**Agent & Algorithms:**
- DQN with experience replay buffer
- Double DQN (reduces Q-value overestimation)
- Dueling DQN architecture (separate value & advantage streams)
- Prioritised Experience Replay (PER) with SumTree
- Noisy Networks for exploration (replaces epsilon-greedy)
- N-step returns for TD targets
- Soft/hard target network updates

**Neural Networks (NumPy only):**
- Configurable MLP with ReLU activations
- Dueling MLP architecture
- SGD optimiser with momentum & gradient clipping
- Adam optimiser with weight decay
- Learning rate schedulers (exponential, step decay)
- NoisyLinear layers for NoisyNet

**Tooling:**
- Matplotlib-based renderer (headless & interactive modes)
- Training script with configurable hyperparameters, logging, CSV export
- Evaluation script with rollout rendering and screenshot export
- Grid search script for hyperparameter tuning
- Comprehensive test suite

## Environment Overview

The agent controls a **red car** (shown as a dot with a heading arrow) that drives on a 2D racetrack. The car uses a kinematic bicycle model — steering rotates its heading, and acceleration changes its speed. The goal is to **complete as many laps as possible** without driving off the road.

Three track types are available, each with a **start/finish line** (green star marker) where the car begins and lap completion is detected. Blue **reward lines** (checkpoint gates) are placed around the track — crossing one gives a bonus reward, encouraging the agent to complete the full loop:

![Environment overview](images/environment_overview.png)

**What the agent observes** (6-dimensional vector):
| Observation | Description |
|---|---|
| `x`, `y` | Car position in world coordinates |
| `heading` | Car orientation in radians |
| `velocity` | Forward speed (0 to max_speed) |
| `dist_to_edge` | Normalized distance to nearest road edge (0 = at edge, 1 = on centerline) |
| `heading_error` | Angle between the car heading and the centerline tangent |

**What the agent can do** (5 discrete actions):
| Action | Steering | Acceleration |
|---|---|---|
| Steer left + accelerate | -0.5 | +1.0 |
| Steer right + accelerate | +0.5 | +1.0 |
| Go straight + accelerate | 0.0 | +1.0 |
| Coast | 0.0 | 0.0 |
| Brake | 0.0 | -0.5 |

**Reward structure:**
- **+0.1** per step while on the track
- **+1.0** for completing a lap
- **+0.5** per reward line (checkpoint gate) crossed
- **-1.0** for driving off the track or colliding with an obstacle (episode ends)

## Quickstart

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint with ruff
uv run ruff check .
```

## Usage

```bash
# Train the DQN agent (default rectangular track)
uv run python scripts/train.py --episodes 300

# Train with circular track, obstacles, and Dueling DQN
uv run python scripts/train.py \
  --track circular \
  --num-obstacles 5 \
  --dueling-dqn \
  --episodes 500

# Train with a JSON config file
uv run python scripts/train.py --config my_config.json

# Evaluate a trained policy
uv run python scripts/evaluate.py --model-path models/best_model.npz

# Grid search over hyperparameters
uv run python scripts/grid_search.py

# Demo the environment with a random policy
uv run python scripts/demo_render_env.py
```

## Training Results

The DQN agent was trained for 100 episodes on the default rectangular track. The training curve below shows the episode reward and average loss.

![Training curve](images/training_curve.png)

After training, the greedy policy was evaluated for 3 episodes. Below are the final frames of each evaluation rollout.

| Episode 1 | Episode 2 | Episode 3 |
|:---------:|:---------:|:---------:|
| ![eval 1](images/eval_ep1_final.png) | ![eval 2](images/eval_ep2_final.png) | ![eval 3](images/eval_ep3_final.png) |

The side-by-side animation below contrasts the **trained DQN policy** (left) with a **random policy** (right) on the same track starting from the same position. The trained agent consistently stays on track and completes laps, while the random agent drives off the track within a few steps.

![Trained vs Random policy comparison](images/trained_vs_random.gif)

## Project constraints

Runtime dependencies:
- numpy
- matplotlib
- pillow (used by renderer for image saving)

Forbidden ML/RL dependencies:
- torch, tensorflow, jax, gymnasium, stable-baselines3

## Roadmap

### ✅ Implemented

1. **Project skeleton** — Python package structure (`pyproject.toml`, `src/`, `tests/`).
2. **2D car physics** — `KinematicCar` / `CarState`: position, heading, velocity, steering, acceleration, speed limits, heading normalisation.
3. **Track representation** — `RectangularTrack`, `CircularTrack`, `Figure8Track` with configurable road width and point-to-segment distance checking.
4. **Racing environment API** — `RacingEnv` with `reset(seed)` / `step(action)` returning `(obs, reward, done, info)`; episode ends when the car leaves the track or collides with an obstacle.
5. **Visualization** — `MatplotlibRenderer`: top-down plot of the track with car position, heading arrow, step counter, and reward overlay. Supports headless mode.
6. **NumPy neural network** — Feed-forward network (`Dense`, ReLU, optional output activation), Dueling MLP, backward pass with NumPy-only gradient descent.
7. **DQN from scratch** — DQN with experience replay, epsilon-greedy, target network, Q-learning loss — all NumPy-only. Includes Double DQN, Dueling DQN, NoisyNet, PER, N-step returns.
8. **Optimisers** — SGD with momentum & gradient clipping, Adam with weight decay.
9. **Training and evaluation scripts** — `scripts/train.py` trains the DQN agent, logs rewards/losses, saves models, and plots training curves. `scripts/evaluate.py` loads a trained model, renders evaluation rollouts, and saves screenshots.
10. **Grid search** — `scripts/grid_search.py` for hyperparameter tuning.
11. **Checkpointing** — Full save/load of model parameters, optimiser state, replay buffer, RNG state, and training metadata.

### 🗺️ Major upgrade roadmap

**Near term:**
- [ ] **Procedural track generation** — Randomised track shapes using splines or bezier curves for greater variety
- [ ] **Side-by-side policy comparison** — `scripts/compare_policies.py` records a GIF contrasting trained vs random policies; could be extended to compare algorithms, seeds, or checkpoints
- [ ] **Higher-quality GIF rendering** — Increase resolution, add legends, speed overlay, and episode info to comparison animations
- [ ] **Ray-casting lidar sensors** — Replace hand-crafted observations with configurable ray-based distance sensors
- [ ] **Continuous action space** — Support for continuous steering & acceleration via policy gradient methods
- [ ] **Frame skipping & action repeat** — Improve training speed and temporal consistency

**Medium term:**
- [ ] **Policy gradient algorithms** — REINFORCE, PPO, and Actor-Critic implementations (NumPy-only)
- [ ] **SAC (Soft Actor-Critic)** — Maximum-entropy RL for continuous control
- [ ] **Multi-agent racing** — Multiple cars on the same track with competitive/collaborative rewards
- [ ] **Video recording** — Save evaluation rollouts as MP4/gif animations
- [ ] **Interactive viewer** — Live rendering window with speed controls

**Long term:**
- [ ] **Model Zoo** — Pre-trained weights for different track/algorithm combinations
- [ ] **Benchmarking harness** — Standardised evaluation across seeds, tracks, and algorithms with statistical reporting
- [ ] **Curriculum learning** — Progressively harder tracks (wider → narrower, simple → complex)
- [ ] **Imitation learning** — Record human/keyboard demonstrations and pre-train via behavioural cloning
- [ ] **Web demo** — WASM-compiled interactive demo (via pyodide or similar)
- [ ] **Documentation site** — Rendered docs with algorithm explanations, equations, and interactive notebooks
