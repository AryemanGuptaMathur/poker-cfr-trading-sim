from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".matplotlib")))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from abstract_game import AbstractGame


def policy_value(
    game: AbstractGame,
    strategy_profile: dict[str, np.ndarray],
    player: int = 0,
    num_episodes: int = 300,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(num_episodes):
        state = game.sample_initial_state(rng)
        values.append(game.policy_value_from_state(state, strategy_profile, player))
    return float(np.mean(values))


def exact_policy_value(
    game: AbstractGame,
    strategy_profile: dict[str, np.ndarray],
    player: int = 0,
) -> float:
    states = game.enumerate_initial_states()
    if states is None:
        raise ValueError(f"{game.name} does not support exact root enumeration")

    return float(
        sum(
            probability * game.policy_value_from_state(state, strategy_profile, player)
            for state, probability in states
        )
    )


def best_response_value(
    game: AbstractGame,
    strategy_profile: dict[str, np.ndarray],
    player: int,
    num_episodes: int = 300,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(num_episodes):
        state = game.sample_initial_state(rng)
        values.append(game.best_response_value_from_state(state, strategy_profile, player))
    return float(np.mean(values))


def exact_best_response_value(
    game: AbstractGame,
    strategy_profile: dict[str, np.ndarray],
    player: int,
) -> float:
    states = game.enumerate_initial_states()
    if states is None:
        raise ValueError(f"{game.name} does not support exact root enumeration")

    return float(
        sum(
            probability * game.best_response_value_from_state(state, strategy_profile, player)
            for state, probability in states
        )
    )


def exploitability(
    game: AbstractGame,
    strategy_profile: dict[str, np.ndarray],
    num_episodes: int = 300,
    seed: int = 0,
) -> float:
    br_0 = best_response_value(game, strategy_profile, player=0, num_episodes=num_episodes, seed=seed)
    br_1 = best_response_value(game, strategy_profile, player=1, num_episodes=num_episodes, seed=seed + 1)
    return float((br_0 + br_1) / 2.0)


def exact_exploitability(
    game: AbstractGame,
    strategy_profile: dict[str, np.ndarray],
    num_episodes: int = 0,
    seed: int = 0,
) -> float:
    del num_episodes, seed
    br_0 = exact_best_response_value(game, strategy_profile, player=0)
    br_1 = exact_best_response_value(game, strategy_profile, player=1)
    return float((br_0 + br_1) / 2.0)


def aggregate_metrics(run_metrics: list[dict[str, list[float]]], key: str = "exploitability") -> dict[str, list[float] | float]:
    if not run_metrics:
        raise ValueError("run_metrics must contain at least one run")

    iterations = run_metrics[0]["iteration"]
    stacked = np.asarray([metrics[key] for metrics in run_metrics], dtype=float)
    final_values = stacked[:, -1]

    return {
        "iteration": [float(iteration) for iteration in iterations],
        "mean": [float(value) for value in np.mean(stacked, axis=0)],
        "std": [float(value) for value in np.std(stacked, axis=0)],
        "final_mean": float(np.mean(final_values)),
        "final_std": float(np.std(final_values)),
    }


def run_benchmark(
    builder: Callable[[int], object],
    seeds: list[int],
    iterations: int,
    eval_every: int,
    eval_episodes: int,
    eval_fn: Callable[..., float] = exploitability,
) -> tuple[dict[str, list[float] | float], list[dict[str, list[float]]], list[object]]:
    runs = []
    solvers = []

    for seed in seeds:
        solver = builder(seed)
        metrics = solver.train(
            iterations=iterations,
            eval_every=eval_every,
            eval_fn=eval_fn,
            eval_episodes=eval_episodes,
        )
        runs.append(metrics)
        solvers.append(solver)

    return aggregate_metrics(runs), runs, solvers


def save_summary(summary: dict[str, object], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def plot_curves(
    curves: dict[str, dict[str, list[float] | float]],
    title: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for label, curve in curves.items():
        iterations = np.asarray(curve["iteration"], dtype=float)
        mean = np.asarray(curve["mean"], dtype=float)
        std = np.asarray(curve["std"], dtype=float)
        plt.plot(iterations, mean, label=label, linewidth=2)
        plt.fill_between(iterations, mean - std, mean + std, alpha=0.18)

    plt.title(title)
    plt.xlabel("Training iteration")
    plt.ylabel("Exploitability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_strategy(
    actions: list[str] | tuple[str, ...],
    strategy: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(actions, strategy)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
