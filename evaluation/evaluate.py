from stable_baselines3 import SAC

from environment.uav_env import UAVEnv
from environment.utils import distance
from config.settings import GOAL_POS

env = UAVEnv()

model = SAC.load("models/sac_uav")

obs, _ = env.reset()

done = False
total_reward = 0

while not done:
    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, done, _, info = env.step(action)

    total_reward += reward

print("Total Reward:", total_reward)
print("Final Position:", env.pos)
print("Goal Position:", GOAL_POS)
print("Distance To Goal:",
      distance(env.pos, GOAL_POS))
print("Energy Left:", env.energy)
print("Reason:",
      info["termination_reason"])