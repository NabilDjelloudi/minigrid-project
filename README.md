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




