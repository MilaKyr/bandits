import os
import json
from typing import List, Optional, Dict, Tuple, NewType, Union
from dataclasses import asdict
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from .base import BaseBandit
from .cascading_bandit import CascadeBanditBase
from .data_structures import DataInformation

StepStats = NewType("RoundStats", List[float])
RoundStats = NewType("RoundStats", List[StepStats])
BanditStats = NewType("BanditStats", Dict[str, RoundStats])

class DatasetHelper:

    def __init__(self, data_info: DataInformation,
                 n_features: int,
                 seed: Optional[int] = None,
                 test_size=0.5,
                 shuffle=True):
        self.data_info = data_info
        self.shuffle = shuffle
        self.n_features = n_features
        self.test_size = test_size
        self.seed = seed if seed else np.random.randint(10_000)
        train_df, self.dataset = self.get_data(data_info, test_size, shuffle)
        self.features = self.prepare_features(train_df, n_features)

    def get_data(self, data_info: DataInformation, test_size: float, shuffle: bool
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(data_info.path, sep=data_info.sep, names=data_info.header)
        if data_info.create:
            assert data_info.rating_column_expression is not None, "Cannot create rating column, no expression provided"
            assert data_info.raw_rating_column_name is not None, "Cannot create rating column, no column name info " \
                                                                 "provided"
            df[data_info.raw_rating_column_name] = (df[data_info.raw_rating_column_name]
                                                    .apply(data_info.rating_column_expression))
        df = pd.pivot_table(df, values=data_info.pivot_info.values,
                            index=[data_info.pivot_info.index],
                            columns=[data_info.pivot_info.columns])
        train_df, test_df = train_test_split(df.fillna(0), test_size=test_size,
                                             random_state=self.seed, shuffle=shuffle)
        return train_df, test_df

    @classmethod
    def prepare_features(cls, df: pd.DataFrame, n_features: int) -> np.ndarray:
        u, s, vt = np.linalg.svd(df)
        vt = vt[:n_features, :]
        vt = normalize(vt, axis=0, norm='l2')
        return vt

    def get_rewards(self, row_number: int, columns: np.array) -> List[int]:
        return self.dataset.iloc[row_number, columns].values

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "name": self.__class__.__name__,
            "data": self.data_info.to_dict(),
            "n_features": self.n_features,
            "seed": self.seed,
            "test_size": self.test_size,
            "shuffle": self.shuffle
        }


class Explorer:
    def __init__(self, data_info: DataInformation,
                 n_features: int,
                 seed=None,
                 test_size=0.5,
                 shuffle=True,
                 save_experiment=True):
        self.helper = DatasetHelper(data_info, n_features, seed, test_size, shuffle)
        self.saver = DataSaver(self.create_experiment_id()) if save_experiment else None

    @classmethod
    def create_experiment_id(cls) -> str:
        return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    def experiment_len(self) -> int:
        return len(self.helper.dataset)

    def get_features(self) -> np.ndarray:
        return self.helper.features

    def experiment(self, bandits: List[BaseBandit], n_rounds: int
                   ) -> Tuple[BanditStats, BanditStats]:
        data_len = self.experiment_len()
        history_stats = defaultdict(list)
        regret_stats = defaultdict(list)
        for round in range(n_rounds):
            for bandit in bandits:
                history, regret = [], data_len
                for step in range(data_len):
                    selected_arms = bandit.get()
                    rewards = self.helper.get_rewards(step, selected_arms)
                    bandit.update(rewards, selected_arms)
                    history.append(max(rewards))
                    regret -= max(rewards)
                history_stats[bandit.name()].append(history)
                regret_stats[bandit.name()].append(regret)
        return history_stats, regret_stats

    def save_experiment(self, bandits: List[CascadeBanditBase],
                        history_stats: BanditStats,
                        regret_stats: BanditStats):
        self.saver.save_bandit_info(bandits, self.to_json())
        self.saver.save_winner_info(regret_stats)
        self.saver.save_plot(history_stats)
        print(f"Done. Data saved in {self.saver.full_path}")

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "name": self.__class__.__name__,
            "data_info": self.helper.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class DataSaver:

    def __init__(self, experiment_id: str):
        self.full_path = self.get_full_path(experiment_id)
        os.makedirs(self.full_path)

    @classmethod
    def mean_cumsum_reward(cls, rewards):
        return np.cumsum(np.mean(rewards, axis=0))

    @classmethod
    def get_parent_folder_abs_path(cls):
        return f"{os.path.abspath(os.getcwd())}"

    def get_full_path(self, experiment_id: str):
        return f"{self.get_parent_folder_abs_path()}/experiments/{experiment_id}"

    def save_bandit_info(self, bandits: List[CascadeBanditBase], explorer_json: str):
        with open(f"{self.full_path}/info.txt", "a") as f:
            f.write(explorer_json + '\n')
            [f.write(bandit.to_json() + '\n') for bandit in bandits]

    def save_winner_info(self, regret_stats: Dict[str, List[float]]):
        with open(f"{self.full_path}/experiment_results.txt", "a") as f:
            for bandit_name, regret_stats in regret_stats.items():
                f.write(f'Mean regret for {bandit_name}:  {np.mean(regret_stats)}' + '\n')

    def save_plot(self, history_stats: Dict[str, List[List[float]]]):
        for bandit_name, bandit_reward_history in history_stats.items():
            x_labels = list(range(len(bandit_reward_history[0])))
            plt.plot(x_labels,
                     self.mean_cumsum_reward(bandit_reward_history),
                     label=bandit_name)
        plt.xlabel('Steps')
        plt.title('Cumulative reward')
        plt.legend()
        plt.savefig(f"{self.full_path}/cumulative_rewards_plot.png")












