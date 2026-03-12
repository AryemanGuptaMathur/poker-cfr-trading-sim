from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from deep_cfr import DeepCFRTrainer
from evaluate import exact_exploitability, exact_policy_value, plot_curves, plot_strategy, run_benchmark, save_summary
from poker_cfr import PokerGame


ITERATIONS = int(os.environ.get("POKER_ITERATIONS", "3000"))
EVAL_EVERY = int(os.environ.get("POKER_EVAL_EVERY", "250"))
EVAL_EPISODES = int(os.environ.get("POKER_EVAL_EPISODES", "400"))
SEEDS = [int(seed) for seed in os.environ.get("POKER_SEEDS", "0,1,2").split(",") if seed]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = RESULTS_DIR / "poker_summary.json"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    curves = {}
    summary: dict[str, object] = {
        "game": "poker",
        "evaluation": "Exact root enumeration over all 6 private-card deals",
        "iterations": ITERATIONS,
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "seeds": SEEDS,
        "trainers": {},
    }
    builders = {
        "Vanilla CFR": lambda seed: PokerGame(seed=seed, variant="vanilla"),
        "CFR+": lambda seed: PokerGame(seed=seed, variant="cfr+"),
        "Deep CFR": lambda seed: DeepCFRTrainer(PokerGame(seed=seed, variant="vanilla"), seed=seed),
    }

    for label, builder in builders.items():
        aggregate, _, solvers = run_benchmark(
            builder=builder,
            seeds=SEEDS,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            eval_fn=exact_exploitability,
        )
        curves[label] = aggregate
        print(f"{label}: final exploitability {aggregate['final_mean']:.4f} +/- {aggregate['final_std']:.4f}")
        exact_values = []
        for solver in solvers:
            game = solver.game if isinstance(solver, DeepCFRTrainer) else solver
            profile = solver.average_strategy_profile()
            exact_values.append(exact_policy_value(game, profile, player=0))
        summary["trainers"][label] = {
            "curve": aggregate,
            "final_exploitability_mean": aggregate["final_mean"],
            "final_exploitability_std": aggregate["final_std"],
            "self_play_value_mean": float(np.mean(exact_values)),
            "self_play_value_std": float(np.std(exact_values)),
        }

    plot_curves(
        curves=curves,
        title="Toy Poker exact exploitability over training",
        output_path=RESULTS_DIR / "poker_exploitability.png",
    )

    representative_solver = PokerGame(seed=0, variant="vanilla")
    representative_solver.train(
        iterations=ITERATIONS,
        eval_every=EVAL_EVERY,
        reset=True,
    )
    profile = representative_solver.average_strategy_profile()
    sample_strategy = profile.get("K_start", np.array([0.5, 0.5], dtype=float))
    plot_strategy(
        actions=["check", "bet"],
        strategy=sample_strategy,
        title="Toy Poker average strategy at K_start",
        output_path=RESULTS_DIR / "poker_strategy.png",
    )
    summary["representative_strategy"] = {
        "infoset": "K_start",
        "actions": ["check", "bet"],
        "average_strategy": sample_strategy.tolist(),
    }

    save_summary(summary, SUMMARY_PATH)
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
