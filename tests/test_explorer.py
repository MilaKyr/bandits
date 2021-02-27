import pytest
import numpy as np
import pandas as pd

from src.explorer import DatasetHelper, Explorer
from src.cascading_bandit import CascadeLinTS
from src.data_structures import PivotInfo, DataInformation


@pytest.fixture
def data_info(tmpdir):
    p = tmpdir.mkdir("sub").join("u.data")
    data = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(10)],
            "item_id": [f"i{i}" for i in range(10)],
            "rating": [i for i in range(10)],
            "timestamp": [i for i in range(10)],
        }
    )
    data.to_csv(p, index=False, header=False)
    return DataInformation(
        path=p,
        sep=",",
        header=["user_id", "item_id", "rating", "timestamp"],
        create=False,
        rating_column_name="rating",
        pivot_info=PivotInfo(values="rating", columns="item_id", index="user_id"),
    )


@pytest.fixture
def explorer(data_info):
    return Explorer(
        data_info, n_features=2, test_size=0.5, seed=1234, save_experiment=False
    )


def test_get_reward(data_info):
    helper = DatasetHelper(data_info, 2, seed=1234)
    result = helper.get_rewards(2, np.array([0, 2]))
    assert np.array_equal(result, helper.dataset.iloc[2, [0, 2]])


def test_explorer_init(explorer):
    assert isinstance(explorer, Explorer)


def test_explorer_dataset_len(explorer):
    assert explorer.experiment_len() == 5


def test_explorer_get_features(explorer):
    assert np.array_equal(explorer.get_features(), explorer.helper.features)


def test_explorer_experiment(explorer, mocker):
    mocker.patch(
        "src.cascading_bandit.CascadeLinTS.get", side_effect=[[1], [1], [1], [1], [1]]
    )
    bandit = CascadeLinTS(explorer.get_features(), max_items=1, n_arms_to_show=1)
    history_stats_result, regret_stats_result = explorer.experiment(
        [bandit], n_rounds=1
    )
    regret_stats_expected_result = {bandit.name(): [4.0]}
    history_stats_expected_result = {bandit.name(): [[0.0, 0.0, 0.0, 1.0, 0.0]]}
    assert history_stats_result == history_stats_expected_result
    assert regret_stats_result == regret_stats_expected_result
