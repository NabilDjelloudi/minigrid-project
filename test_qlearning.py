import gymnasium as gym
import numpy as np

# Importation de l'environnement personnalisé
from minigrid.envs.gotodoor import GoToDoorEnv  

# Enregistrement de l'environnement
gym.register(
    id="MiniGrid-GoToDoor-5x5-v0",
    entry_point="minigrid.envs.gotodoor:GoToDoorEnv",
    max_episode_steps=500,
)

# Charger la Q-table entraînée avec Q-learning
q_table = np.load("q_table_qlearning.npy", allow_pickle=True).item()

# Création de l'environnement
env = gym.make("MiniGrid-GoToDoor-5x5-v0", render_mode=None)
env = env.unwrapped

# Paramètres de test
test_episodes = 100  # Nombre d'épisodes de test
success_count = 0

# Fonction pour obtenir l'état de l'agent
def get_agent_state():
    """Retourne l'état actuel de l'agent."""
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    target_color = env.mission.split()[-2]  # Extraire la couleur de la mission
    return (agent_pos[0], agent_pos[1], agent_dir, target_color)

# Exécution des tests
for episode in range(test_episodes):
    obs, info = env.reset()

    # Fixer la mission pour toujours aller à la porte jaune
    env.target_color = 'blue'
    env.mission = f"go to the blue door"

    state = get_agent_state()
    done = False
    step_count = 0
    success = False
    total_reward = 0

    while not done and step_count < 200:
        # Sélectionner la meilleure action à partir de la Q-table
        if state in q_table:
            action = max(q_table[state], key=q_table[state].get)  # Action avec la plus grande valeur
        else:
            action = env.action_space.sample()  # Action aléatoire si l'état est inconnu

        next_obs, reward, terminated, truncated, info = env.step(action)
        state = get_agent_state()
        total_reward += reward
        done = terminated or truncated
        step_count += 1

        # Vérifier si la récompense positive est obtenue
        if reward > 0:
            success = True
            break

    if success:
        success_count += 1

    print(f"Test {episode + 1}/{test_episodes} - Récompense totale : {total_reward:.2f}")

# Calculer le taux de réussite
success_rate = (success_count / test_episodes) * 100
print(f"\nSuccès: {success_count}/{test_episodes}")
print(f"Taux de réussite: {success_rate:.2f}%")

# Fermer l'environnement
env.close()
