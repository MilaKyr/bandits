import pytest
import numpy as np

from src.cascading_bandit import CascadeLinTS


@pytest.fixture
def cascade_lin_ts_bandit():
    features = np.array(
        [
            [1.0, 6.0, 11.0],
            [2.0, 7.0, 12.0],
            [3.0, 8.0, 13.0],
            [4.0, 9.0, 14.0],
            [5.0, 10.0, 15.0],
        ]
    )
    return CascadeLinTS(features, max_items=2, n_arms_to_show=2)


def test_get_n_features(cascade_lin_ts_bandit):
    assert cascade_lin_ts_bandit.n_features == 5


def test_get_and_save_top_items(cascade_lin_ts_bandit):
    scored_items = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = cascade_lin_ts_bandit.get_and_save_top_items(scored_items)
    expected_result = np.array([4.0, 3.0])
    assert np.array_equal(result, expected_result)


def test_update_vector_b(cascade_lin_ts_bandit, mocker):
    mocker.patch("src.cascading_bandit.CascadeLinTS.get", return_value=np.array([2, 1]))
    rewards = [0, 1]
    cascade_lin_ts_bandit.update(rewards, np.array([2, 1]))
    assert np.array_equal(
        cascade_lin_ts_bandit.vector_b, np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    )


def test_vector_b_and_matrix_m_initialization(cascade_lin_ts_bandit):
    assert np.array_equal(cascade_lin_ts_bandit.vector_b, np.zeros(5))
    assert np.array_equal(cascade_lin_ts_bandit.matrix_m, np.identity(5))


def test_name(cascade_lin_ts_bandit):
    assert cascade_lin_ts_bandit.name() == "CascadeLinTS"


def test_full_name(cascade_lin_ts_bandit):
    assert cascade_lin_ts_bandit.full_name() == "CascadeLinTS_1"
