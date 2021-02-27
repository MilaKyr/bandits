from typing import List, Union
from abc import ABC, abstractmethod

import numpy as np


class BaseBandit(ABC):

    def __init__(self, features: np.ndarray):
        self.n_features = self._get_n_features(features)
        self.features = features

    @classmethod
    def _get_n_features(cls, features: np.ndarray) -> int:
        return features.shape[0]

    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get(self) -> Union[int, List[int]]:   # pragma: no cover
        pass

    @abstractmethod
    def update(self, rewards: List[int], last_selected_arms: np.array) -> None:   # pragma: no cover
        pass
