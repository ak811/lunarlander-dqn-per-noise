# LunarLander DQN: Prioritized Replay + Parameter Noise

This repository implements a **DQN-based agent** for `LunarLander-v3`, with two knobs that actually matter:

1. **Experience Replay**: Prioritized Experience Replay (PER) vs Uniform Replay vs Online updates
2. **Exploration**: **Parameter Noise** (non-ε-greedy exploration) with optional ε-greedy baseline

It logs training metrics, saves checkpoints, plots reward curves, and records videos when a new best policy appears.

---

## Environment

`LunarLander-v3` (Gymnasium Box2D):

- Observation: 8D continuous state
- Actions: 4 discrete thrusters
- Reward shaping encourages stable landing, penalizes crashes and fuel waste

---

## What’s implemented

### Experience Replay
- **Uniform replay**: sample transitions uniformly from a replay buffer.
- **PER**: sample transitions proportional to priority (|TD error|), with importance-sampling weights and β annealing.
- **Online**: no replay buffer; update every step (baseline).

### Exploration
- **Parameter noise**: Gaussian noise applied to network parameters to create a perturbed policy for an episode.
- `epsilon_greedy` is included as a baseline (because reviewers love baselines).

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Default training
```bash
python scripts/train.py --config configs/default.yaml
```

### Override config values from CLI
```bash
python scripts/train.py --config configs/default.yaml --set replay_mode=uniform exploration_mode=none
```

Common overrides:
- `replay_mode=per|uniform|online`
- `exploration_mode=param_noise|epsilon_greedy|none`
- `noise_scale=0.1`
- `num_episodes=800`

---

## Ablations

Run the 4-way comparison (uniform/PER × noise/no-noise):

```bash
python scripts/ablation.py --base configs/default.yaml --ablation configs/ablation.yaml
```

This produces separate run folders under `experiments/` with plots and models.

---

## Evaluation

```bash
python scripts/evaluate.py   --config experiments/<run>/config.yaml   --checkpoint experiments/<run>/models/best_model.pth   --episodes 20
```

---

## Video recording

```bash
python scripts/record_video.py   --config experiments/<run>/config.yaml   --checkpoint experiments/<run>/models/best_model.pth   --out_dir assets/videos   --name best_lunarlander_dqn
```

---

## Testing

```bash
pytest -q
```

---

## License

MIT (See `LICENSE`).
