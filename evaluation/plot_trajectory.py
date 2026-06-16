import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

from environment.uav_env import UAVEnv
from config.settings import *
from environment.map_data import obstacles, users

env = UAVEnv()

model = SAC.load("models/sac_uav")

obs, _ = env.reset()

done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

path = np.array(env.path)

plt.figure(figsize=(8, 8))

plt.plot(path[:, 0], path[:, 1], linewidth=2, label="UAV Path")

plt.scatter(
    START_POS[0],
    START_POS[1],
    marker="o",
    s=100,
    label="Start"
)

plt.scatter(
    GOAL_POS[0],
    GOAL_POS[1],
    marker="*",
    s=200,
    label="Goal"
)

for ox, oy, r in obstacles:
    circle = plt.Circle((ox, oy), r, fill=False)
    plt.gca().add_patch(circle)

for u in users:
    plt.scatter(u[0], u[1], marker="x", s=80)

plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)

plt.grid(True)
plt.legend()

plt.savefig("trajectory.png", dpi=300)

plt.show()