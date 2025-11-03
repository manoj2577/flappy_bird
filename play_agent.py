from stable_baselines3 import PPO
from flappy import FlappyBirdEnv

env = FlappyBirdEnv()
model = PPO.load("flappybird_ppo", env=env)

while True:
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
