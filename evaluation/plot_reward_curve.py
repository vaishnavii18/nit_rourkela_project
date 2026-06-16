import os
import pandas as pd
import matplotlib.pyplot as plt

# Find the CSV file automatically
log_file = None

for file in os.listdir("logs"):
    if file.endswith(".csv"):
        log_file = os.path.join("logs", file)
        break

if log_file is None:
    raise FileNotFoundError("No CSV file found inside logs folder")

print("Using log file:", log_file)

df = pd.read_csv(log_file, skiprows=1)

print(df.head())

rewards = df["r"]

smoothed = rewards.rolling(
    window=20,
    min_periods=1
).mean()

plt.figure(figsize=(10, 5))

plt.plot(smoothed)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")

plt.grid(True)

plt.savefig("reward_curve.png", dpi=300)

plt.show()

print("reward_curve.png saved successfully")