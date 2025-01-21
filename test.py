import gymnasium as gym
# Enregistrer l'environnement dans Gym
from minigrid.envs.gotodoor import GoToDoorEnv  # Assurez-vous que c'est le bon chemin d'import

# Enregistrement de l'environnement dans Gym
gym.register(
    id="MiniGrid-GoToDoor-5x5-v0",
    entry_point="minigrid.envs.gotodoor:GoToDoorEnv",  # L'emplacement correct de ta classe GoToDoorEnv
    max_episode_steps=1000,  # Nombre maximum de steps
)
env = gym.make("MiniGrid-GoToDoor-5x5-v0")
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Effectuer une action aléatoire
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()
