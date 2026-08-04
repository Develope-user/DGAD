from typing import List, Tuple

import numpy as np
from pyquaternion import Quaternion


class CustomNuscenesBox:
    """Minimal nuScenes box used for result coordinate conversion."""

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 fut_trajs: List[float],
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        if np.any(np.isnan(center)) or np.any(np.isnan(size)):
            raise ValueError('Box center and size must not contain NaN values')
        if len(center) != 3 or len(size) != 3:
            raise ValueError('Box center and size must contain three values')
        if not isinstance(orientation, Quaternion):
            raise TypeError('orientation must be a Quaternion')

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token
        self.fut_trajs = np.array(fut_trajs)

    def translate(self, translation: np.ndarray) -> None:
        self.center += translation

    def rotate(self, quaternion: Quaternion) -> None:
        self.center = quaternion.rotation_matrix @ self.center
        self.orientation = quaternion * self.orientation
        self.velocity = quaternion.rotation_matrix @ self.velocity
