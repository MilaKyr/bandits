from abc import abstractmethod
from typing import List, Union, Tuple, Dict

import json
import numpy as np

from .base import BaseBandit


class CascadeBanditBase(BaseBandit):

    def __init__(self, features: np.ndarray,
                 variance: float,
                 n_arms_to_show: int,
                 max_items: int):
        super().__init__(features)
        self.n_arms_to_show = n_arms_to_show
        self.max_items = max_items
        self.variance = variance
        self.vector_b, self.matrix_m = self.initialize_vector_b_and_matrix_m()

    def full_name(self):
        return f"{self.name()}_{self.variance}"

    def initialize_vector_b_and_matrix_m(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_features), np.identity(self.n_features)

    def get(self) -> np.ndarray:
        return self.get_and_save_top_items(self.get_item_scores())

    @abstractmethod
    def get_item_scores(self) -> np.ndarray:  # pragma: no cover
        pass

    def get_and_save_top_items(self, scored_items: np.ndarray) -> np.ndarray:
        return scored_items.argsort()[-self.n_arms_to_show:][::-1]

    def get_previous_b_and_m(self) -> Tuple[np.ndarray, np.ndarray]:
        inv_matrix_m = np.linalg.pinv(self.matrix_m)
        prev_theta = (self.variance ** -2) * inv_matrix_m @ self.vector_b
        return prev_theta, inv_matrix_m

    def update(self, rewards: List[int], last_selected_arms: np.array) -> None:
        if np.sum(rewards) > 0:
            first_reward = next(iter(np.flatnonzero(rewards)))
            rewards = rewards[:first_reward + 1]
            last_selected_arms = last_selected_arms[:first_reward + 1]

        for arm_id, arm in enumerate(last_selected_arms):
            self.update_arm(arm, rewards[arm_id])

    def update_arm(self, arm: int, reward: int) -> None:
        x = self.features.T[arm, :]
        self.matrix_m += self.variance ** -2 * x @ x.T
        self.vector_b += x * reward

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "name": self.name(),
            "variance": self.variance,
            "n_arms_to_show": self.n_arms_to_show,
            "max_items": self.max_items
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class CascadeLinTS(CascadeBanditBase):

    def __init__(self, features: np.ndarray,
                 variance: float = 1,
                 n_arms_to_show: int = 4,
                 max_items: int = 16):
        super().__init__(features, variance, n_arms_to_show, max_items)

    def get_item_scores(self) -> np.ndarray:
        return self.features.T @ self.sample_theta(*self.get_previous_b_and_m())

    def sample_theta(self, previous_theta, inverse_identity_feature_matrix) -> np.ndarray:
        try:
            return np.random.multivariate_normal(previous_theta, inverse_identity_feature_matrix)
        except np.linalg.LinAlgError as e:
            return np.random.multivariate_normal(*self.initialize_vector_b_and_matrix_m())


class CascadeLinUCB(CascadeBanditBase):

    def __init__(self, features: np.ndarray,
                 c: int,
                 variance: float = 1,
                 n_arms_to_show: int = 4,
                 max_items: int = 16):
        super().__init__(features, variance, n_arms_to_show, max_items)
        self.c = c

    def full_name(self) -> str:
        return f"{self.name()}_{self.variance}_{self.c}"

    def get_item_scores(self) -> List[float]:
        prev_theta, inv_matrix_m = self.get_previous_b_and_m()
        prediction = self.features.T @ prev_theta
        upper_confident_bound = (self.features.T @ inv_matrix_m * self.features.T).sum(1)
        all_items = prediction + self.c * np.sqrt(upper_confident_bound)
        all_items[all_items > 1] = 1
        return all_items

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "name": self.name(),
            "c": self.c,
            "variance": self.variance,
            "n_arms_to_show": self.n_arms_to_show,
            "max_items": self.max_items
        }


