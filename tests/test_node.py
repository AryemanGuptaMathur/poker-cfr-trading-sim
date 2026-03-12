import numpy as np

from utils import Node


def test_regret_matching_uses_only_positive_regrets():
    node = Node(["a", "b", "c"])
    node.regret_sum = np.array([3.0, -2.0, 1.0], dtype=float)

    strategy = node.get_strategy()

    assert np.allclose(strategy, np.array([0.75, 0.0, 0.25]))


def test_average_strategy_defaults_to_uniform_when_unvisited():
    node = Node(["left", "right"])

    average_strategy = node.get_average_strategy()

    assert np.allclose(average_strategy, np.array([0.5, 0.5]))


def test_cfr_plus_clips_negative_regrets():
    node = Node(["left", "right"])
    node.update_regrets(
        opponent_reach=1.0,
        instantaneous_regrets=np.array([-1.0, 2.0]),
        variant="cfr+",
    )

    assert np.allclose(node.regret_sum, np.array([0.0, 2.0]))
