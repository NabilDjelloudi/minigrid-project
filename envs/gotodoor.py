from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv
import random



class GoToDoorEnv(MiniGridEnv):
    """
    ## Description

    This environment is a room with four doors, one on each wall. The agent
    receives a textual (mission) string as input, telling it which door to go
    to, (eg: "go to the red door"). It receives a positive reward for performing
    the `done` action next to the correct door, as indicated in the mission
    string.

    ## Mission Space

    "go to the {color} door"

    {color} is the color of the door. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | Unused               |
    | 4   | drop         | Unused               |
    | 5   | toggle       | Unused               |
    | 6   | done         | Done completing task |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent stands next the correct door performing the `done` action.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-GoToDoor-5x5-v0`
    - `MiniGrid-GoToDoor-6x6-v0`
    - `MiniGrid-GoToDoor-8x8-v0`

    """

    def __init__(self, size=5, max_steps: int | None = None, **kwargs):
        assert size == 5  # Taille fixe à 5x5
        self.size = size
        if max_steps is None:
            max_steps = 4 * size**2  # Ajuster le nombre d'étapes
        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission, ordered_placeholders=[COLOR_NAMES]),
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs
        )


    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} door"

    def _gen_grid(self, width, height):
            # Créer la grille vide
            self.grid = Grid(width, height)

            # Ajouter les murs autour de la grille
            self.grid.wall_rect(0, 0, width, height)

            # Définir les positions des portes à l'extérieur de la grille
            doorPos = [
                (width // 2, 0),           # Porte en haut (milieu du haut)
                (width // 2, height - 1),   # Porte en bas (milieu du bas)
                (0, height // 2),           # Porte à gauche (milieu gauche)
                (width - 1, height // 2)    # Porte à droite (milieu droite)
            ]

            # Choisir aléatoirement des couleurs parmi les options disponibles
            doorColors = random.sample(["blue", "green", "grey", "purple", "red", "yellow"], len(doorPos))

            # Ajouter les portes à la grille avec des couleurs aléatoires
            for idx, pos in enumerate(doorPos):
                color = doorColors[idx]
                self.grid.set(*pos, Door(color))

            # Placer l'agent au centre de la grille
            self.agent_pos = (width // 2, height // 2)
            self.agent_dir = 0  # Direction initiale vers la droite

            # Sélectionner une porte cible aléatoire
            doorIdx = self._rand_int(0, len(doorPos))
            self.target_pos = doorPos[doorIdx]
            self.target_color = doorColors[doorIdx]

            # Générer le message de mission avec la nouvelle couleur de la porte cible
            self.mission = f"go to the {self.target_color} door"

    def step(self, action):
        if action in [3, 4, 5]:  # Ignorer les actions inutiles
            reward = -0.2  # Petite pénalité pour éviter ces actions
            return self.gen_obs(), reward, False, False, {}

        obs, reward, terminated, truncated, info = super().step(action)
        
        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Vérifier si l'agent est à côté de la bonne porte
        if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
            reward = self._reward()
            self.reset()  # Respawn automatique après succès
            terminated = False  # Ne pas terminer l'épisode

        return obs, reward, terminated, truncated, info
