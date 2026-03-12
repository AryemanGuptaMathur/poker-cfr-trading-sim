import numpy as np

from evaluate import exact_exploitability
from mccfr import ExternalSamplingMCCFRTrainer
from poker_cfr import PokerGame


def test_mccfr_trains_and_returns_metrics():
    trainer = ExternalSamplingMCCFRTrainer(PokerGame(seed=0), seed=0)

    metrics = trainer.train(iterations=5, eval_every=5, eval_fn=exact_exploitability, eval_episodes=0)

    assert metrics["iteration"] == [1.0, 5.0]
    assert len(metrics["exploitability"]) == 2


def test_mccfr_average_strategy_is_valid_distribution():
    trainer = ExternalSamplingMCCFRTrainer(PokerGame(seed=0), seed=0)
    trainer.train(iterations=10, eval_every=10, reset=True)

    profile = trainer.average_strategy_profile()
    strategy = profile["K_start"]

    assert np.isclose(np.sum(strategy), 1.0)
    assert np.all(strategy >= 0.0)
