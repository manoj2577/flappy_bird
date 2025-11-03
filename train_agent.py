from stable_baselines3 import PPO
from flappy import FlappyBirdEnv

env = FlappyBirdEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)

model.save("flappybird_ppo")
env.close()
