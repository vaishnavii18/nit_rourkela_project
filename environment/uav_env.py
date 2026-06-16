import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.settings import *
from environment.utils import *
from environment.map_data import *


class UAVEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-MAX_ACCEL, -np.pi], dtype=np.float32),
            high=np.array([MAX_ACCEL, np.pi], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = START_POS.astype(np.float32).copy()

        self.velocity = np.array(
            [0.0, 0.0],
            dtype=np.float32
        )

        self.energy = MAX_ENERGY

        self.path = []

        self.prev_distance = distance(
            self.pos,
            GOAL_POS
        )

        return self.state(), {}

    def state(self):
        goal_dist = distance(self.pos, GOAL_POS)

        return np.array([
            self.pos[0] / AREA_SIZE,
            self.pos[1] / AREA_SIZE,

            (self.velocity[0] + MAX_SPEED) / (2 * MAX_SPEED),
            (self.velocity[1] + MAX_SPEED) / (2 * MAX_SPEED),

            self.energy / MAX_ENERGY,

            goal_dist / AREA_SIZE,

            BS_POS[0] / AREA_SIZE,
            BS_POS[1] / AREA_SIZE,

            INTERFERENCE_POS[0] / AREA_SIZE,
            INTERFERENCE_POS[1] / AREA_SIZE,

            0.0,
            0.0
        ], dtype=np.float32)

    def total_throughput(self):
        total = 0

        for u in users:

            d_user = distance(self.pos, u)
            d_bs = distance(self.pos, BS_POS)
            d_int = distance(self.pos, INTERFERENCE_POS)

            signal = channel_gain(d_user)
            relay = channel_gain(d_bs)
            interference = channel_gain(d_int)

            sinr = signal / (interference + 1e-9)

            total += throughput(sinr)

        return total

    def collision(self):

        for ox, oy, r in obstacles:

            d = np.sqrt(
                (self.pos[0] - ox) ** 2 +
                (self.pos[1] - oy) ** 2
            )

            if d < r:
                return True

        return False

    def step(self, action):

        accel = float(action[0])
        angle = float(action[1])

        self.velocity += np.array([
            accel * np.cos(angle),
            accel * np.sin(angle)
        ])

        speed = np.linalg.norm(self.velocity)

        max_speed = 5.0

        if speed > max_speed:
            self.velocity = (
                self.velocity / speed
            ) * max_speed
            speed = max_speed

        self.pos += self.velocity

        self.path.append(self.pos.copy())

        self.energy -= speed * 2

        goal_distance = distance(
            self.pos,
            GOAL_POS
        )

        throughput_reward = (
            self.total_throughput() / 1e6
        )

        reward = throughput_reward

        progress = (
            self.prev_distance -
            goal_distance
        )

        reward += progress * 10

        self.prev_distance = goal_distance

        done = False
        termination_reason = "Running"

        if self.collision():
            reward -= 100
            done = True
            termination_reason = "Collision"

        if self.energy <= 0:
            reward -= 100
            done = True
            termination_reason = "Energy Exhausted"

        if goal_distance < 20:
            reward += 1000
            done = True
            termination_reason = "Goal Reached"

        if (
            self.pos[0] < 0 or
            self.pos[0] > AREA_SIZE or
            self.pos[1] < 0 or
            self.pos[1] > AREA_SIZE
        ):
            reward -= 500
            done = True
            termination_reason = "Out of Bounds"

        return (
            self.state(),
            reward,
            done,
            False,
            {
                "termination_reason":
                termination_reason
            }
        )