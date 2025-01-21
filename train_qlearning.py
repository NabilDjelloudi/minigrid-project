import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importation de l'environnement personnalisé
from minigrid.envs.gotodoor import GoToDoorEnv  

# Enregistrement de l'environnement
gym.register(
    id="MiniGrid-GoToDoor-5x5-v0",
    entry_point="minigrid.envs.gotodoor:GoToDoorEnv",
    max_episode_steps=500,
)

# Création de l'environnement
env = gym.make("MiniGrid-GoToDoor-5x5-v0", render_mode=None)
env = env.unwrapped

# Paramètres du liens
alpha = 0.5  # Taux d'apprentissage
gamma = 0.5  # Facteur de discount
epsilon = 0.1  # Probabilité d'exploration
episodes = 1000  # Nombre d'épisodes d'entraînement

# Actions utiles uniquement (exclusion des actions inutiles)
actions = [0, 1, 2, 6]  # left, right, forward, done

# Initialisation de la Q-table
q_table = {}
visit_count = {}  # Dictionnaire pour suivre le nombre de visites par état

def update_visit_count(state):
    """Mettre à jour le compteur de visites d'un état et appliquer une exploration forcée."""
    if state not in visit_count:
        visit_count[state] = 0
    visit_count[state] += 1

def initialize_q_table():
    """Initialisation de la Q-table avec des valeurs aléatoires pour chaque état valide."""
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            for direction in range(4):  # 4 directions (0: droite, 1: bas, 2: gauche, 3: haut)
                for color in ["green", "yellow", "blue", "purple"]:
                    q_table[(x, y, direction, color)] = {a: np.random.uniform(0, 1) for a in actions}

def get_agent_state():
    agent_pos = env.agent_pos  # Position de l'agent
    agent_dir = env.agent_dir  # Direction de l'agent
    target_color = env.mission.split()[-2]  # Extraire la couleur de la mission
    return (agent_pos[0], agent_pos[1], agent_dir, target_color)

def epsilon_greedy_policy(state, epsilon):
    """Politique epsilon-greedy pour choisir une action valide."""
    if state not in q_table:
        q_table[state] = {a: np.random.uniform(0, 1) for a in actions}
        
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Exploration
    else:
        return max(q_table[state], key=q_table[state].get)  # Exploitation

def epsilon_decay(episode, epsilon, decay_rate=0.999, min_epsilon=0.05):
    return max(min_epsilon, epsilon * (decay_rate ** episode))

# Initialisation de la Q-table
initialize_q_table()

# Variables de suivi des performances
reward_history = []
success_count, fail_count = 0, 0

for episode in range(episodes):
    obs, info = env.reset()
    state = get_agent_state()
    total_reward = 0
    current_epsilon = epsilon_decay(episode, epsilon)
    done = False
    step_count = 0

    while not done and step_count < 200:
        action = epsilon_greedy_policy(state, current_epsilon)
        update_visit_count(state)  # Mise à jour du compteur de visites
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Affichage de l'environnement toutes les 50 étapes
        if step_count % 50 == 0:
            env.render()
        next_state = get_agent_state()

        # Ajouter pénalité si bloqué au même état
        if state == next_state:
            reward -= 0.1  # Réduire la pénalité

        # Encourager le rapprochement à la porte cible
        distance_before = np.linalg.norm(np.array(state[:2]) - np.array(env.target_pos))
        distance_after = np.linalg.norm(np.array(next_state[:2]) - np.array(env.target_pos))

        if distance_after < distance_before:
            reward += 1.0  # Récompense pour se rapprocher
        else:
            reward -= 0.05  # Pénalité si s'éloigne

        # Normalisation de la récompense dans l'intervalle [0, 1]
        reward = max(0, min(1, reward))

        # Vérifier si l'état suivant est dans la Q-table, sinon l'initialiser
        if next_state not in q_table:
            q_table[next_state] = {a: np.random.uniform(0, 1) for a in actions}

        # Q-learning mise à jour de la Q-table
        best_next_action = max(q_table[next_state], key=q_table[next_state].get)
        q_table[state][action] += alpha * (
            reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
        )

        state = next_state
        total_reward += reward
        done = terminated or truncated
        step_count += 1

    # Normalisation de la récompense totale pour la placer entre [0, 1]
    normalized_total_reward = total_reward / 100  # Facteur de normalisation
    normalized_total_reward = max(0, min(1, normalized_total_reward))

    reward_history.append(normalized_total_reward)

    if terminated:
        success_count += 1
    else:
        fail_count += 1

    # Arrêt anticipé si 80% de succès après au moins 10 épisodes réussis
    if success_count / (success_count + fail_count) >= 0.8 and success_count > 10:
        print("Objectif atteint, arrêt anticipé.")
        break

    if (episode + 1) % 50 == 0:
        print(f"Épisode {episode + 1}/{episodes} - Récompense totale : {normalized_total_reward:.2f}")
        max_q_value = max([max(q_table[s].values()) for s in q_table])
        print(f"Épisode {episode + 1}: Max valeur Q {max_q_value:.4f}")

# Enregistrement des récompenses pour analyse
df = pd.DataFrame({'episode': range(1, len(reward_history) + 1), 'reward': reward_history})
df.to_csv('rewards_qlearning.csv', index=False)
print("Historique des récompenses sauvegardé dans 'rewards_qlearning.csv'.")

# Tracer l'évolution des récompenses
plt.plot(reward_history)
plt.xlabel('Épisodes')
plt.ylabel('Récompense totale ')
plt.title("Évolution des récompenses avec Q-learning")
plt.show()

# Sauvegarde de la Q-table pour une utilisation future
np.save("q_table_qlearning.npy", q_table)
print("Q-table sauvegardée sous 'q_table_qlearning.npy'.")

# Fermer l'environnement
env.close()
