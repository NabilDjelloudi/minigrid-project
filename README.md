# MiniGrid Project - SARSA & Q-Learning Training

Ce projet implémente l'apprentissage par renforcement (SARSA et Q-Learning) sur l'environnement **MiniGrid GoToDoor**, dans lequel un agent doit se rendre à une porte spécifique en fonction d'une instruction textuelle.

## Structure du projet

Le projet contient les fichiers suivants :

- `train_sarsa.py` - Script pour entraîner l'agent avec l'algorithme SARSA.
- `train_qlearning.py` - Script pour entraîner l'agent avec l'algorithme Q-Learning.
- `test_sarsa.py` - Script pour tester l'agent entraîné avec SARSA sur des scénarios spécifiques.
- `test_qlearning.py` - Script pour tester l'agent entraîné avec Q-Learning sur des scénarios spécifiques.
- `gotodoor.py` - Définition de l'environnement personnalisé GoToDoor.
- `rewards_sarsa.csv` - Historique des récompenses SARSA (généré après entraînement).
- `rewards_qlearning.csv` - Historique des récompenses Q-Learning (généré après entraînement).
- `q_table_sarsa.npy` - Q-table sauvegardée après l'entraînement SARSA.
- `q_table_qlearning.npy` - Q-table sauvegardée après l'entraînement Q-Learning.

---

## Prérequis

Avant d'exécuter le projet, assurez-vous d'avoir installé les dépendances suivantes :

```bash
pip install gymnasium minigrid numpy matplotlib pandas
Exécution des scripts

```

## 1. Entraînement avec SARSA
Pour entraîner l'agent avec l'algorithme SARSA, exécutez :
python train_sarsa.py
Cela entraînera l'agent et sauvegardera la Q-table dans q_table_sarsa.npy.

## 2. Test de l'agent SARSA
Pour tester l'agent SARSA sur l'environnement GoToDoor, exécutez :
python test_sarsa.py
Cela évaluera l'agent sur plusieurs épisodes et affichera le taux de réussite.

## 3. Entraînement avec Q-Learning
Pour entraîner l'agent avec l'algorithme Q-Learning, exécutez :
python train_qlearning.py
Cela entraînera l'agent et sauvegardera la Q-table dans q_table_qlearning.npy.

## 4. Test de l'agent Q-Learning
Pour tester l'agent Q-Learning sur l'environnement GoToDoor, exécutez :
python test_qlearning.py
Cela évaluera l'agent sur plusieurs épisodes et affichera le taux de réussite.

## 5. Affichage de l'agent
Pour afficher visuellement l'agent dans la grille pendant l'entraînement, modifiez la ligne suivante dans train_sarsa.py et train_qlearning.py :

env = gym.make("MiniGrid-GoToDoor-5x5-v0", render_mode='human')  # Pour affichage visuel
Si vous ne souhaitez pas d'affichage, utilisez :

env = gym.make("MiniGrid-GoToDoor-5x5-v0", render_mode=None)  # Sans affichage

## Structure de l'environnement GoToDoor
# L'environnement personnalisé GoToDoor contient :

Une grille 5x5 entourée de murs.
Quatre portes positionnées aux centres des murs.
Un agent qui reçoit la mission d'aller vers une porte d'une couleur spécifique.
Récompenses normalisées entre '[0, 1]' selon la distance et le succès.

## Résultats
Les performances de l'agent sont enregistrées dans les fichiers CSV respectifs (rewards_sarsa.csv et rewards_qlearning.csv) et peuvent être visualisées à l'aide de matplotlib :
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rewards_sarsa.csv')
plt.plot(df['episode'], df['reward'])
plt.xlabel('Épisode')
plt.ylabel('Récompense')
plt.title('Évolution des récompenses SARSA')
plt.show()




