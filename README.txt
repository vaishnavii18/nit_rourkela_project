# UAV Path Planning using Soft Actor-Critic (SAC)

## Overview

This project implements a Reinforcement Learning based UAV path planning framework using the Soft Actor-Critic (SAC) algorithm.

### Features

- UAV navigation in a 2D environment
- Obstacle avoidance
- Throughput-aware reward function
- Energy constraints
- SAC training using Stable-Baselines3

## Project Structure

```text
UAV_RL_Project/
├── config/
├── environment/
├── training/
├── evaluation/
├── models/
├── logs/
└── main.py
```

## Training

```bash
python main.py
```

## Evaluation

```bash
python -m evaluation.evaluate
```

## Results

- Goal Reached
- Energy Efficient Navigation
- SAC-based Continuous Control