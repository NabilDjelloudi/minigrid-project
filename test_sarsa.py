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

# Charger la Q-table entraînée
q_table = np.load("q_table_sarsa.npy", allow_pickle=True).item()

# Création de l'environnement
env = gym.make("MiniGrid-GoToDoor-5x5-v0", render_mode=None)
env = env.unwrapped

# Paramètres de test
test_episodes = 100  # Nombre d'épisodes de test
success_count = 0

# Fonction pour obtenir l'état de l'agent
def get_agent_state():
    """Retourne l'état de l'agent sous forme d'un tuple."""
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    target_color = env.mission.split()[-2]  # Extraire la couleur de la mission
    return (agent_pos[0], agent_pos[1], agent_dir, target_color)

# Exécution du test
for episode in range(test_episodes):
    obs, info = env.reset()

    # Fixer la mission pour toujours aller à la porte jaune
    env.target_color = 'yellow'
    env.mission = f"go to the yellow door"
    
    state = get_agent_state()
    done = False
    step_count = 0
    success = False

    while not done and step_count < 200:
        # Sélectionner la meilleure action à partir de la Q-table
        if state in q_table:
            action = max(q_table[state], key=q_table[state].get)
        else:
            action = env.action_space.sample()  # Action aléatoire si état inconnu

        next_obs, reward, terminated, truncated, info = env.step(action)
        state = get_agent_state()
        done = terminated or truncated
        step_count += 1

        # Vérifier si la mission est accomplie
        if reward > 0:
            success = True
            break

    if success:
        success_count += 1

    print(f"Test {episode + 1}/{test_episodes} - Récompense totale : {reward:.2f}")

# Calculer le taux de réussite
success_rate = (success_count / test_episodes) * 100
print(f"\nSuccès: {success_count}/{test_episodes}")
print(f"Taux de réussite: {success_rate:.2f}%")
