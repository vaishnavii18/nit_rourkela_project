from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from environment.uav_env import UAVEnv

def train():
    env = UAVEnv()

    # Record training statistics
    env = Monitor(env, "logs")

    model = SAC(
             "MlpPolicy",
             env,
             learning_rate=3e-4,
             batch_size=256,
             verbose=1
      )

    model.learn(total_timesteps=100000)

    model.save("models/sac_uav")

    print("Training Complete")