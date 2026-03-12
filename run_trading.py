from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from deep_cfr import DeepCFRTrainer
from evaluate import exploitability, policy_value, plot_curves, plot_strategy, run_benchmark, save_summary
from mccfr import ExternalSamplingMCCFRTrainer
from trading_sim import MarketMakingGame


ITERATIONS = int(os.environ.get("TRADING_ITERATIONS", "2000"))
EVAL_EVERY = int(os.environ.get("TRADING_EVAL_EVERY", "200"))
EVAL_EPISODES = int(os.environ.get("TRADING_EVAL_EPISODES", "300"))
SEEDS = [int(seed) for seed in os.environ.get("TRADING_SEEDS", "0,1,2").split(",") if seed]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = RESULTS_DIR / "trading_summary.json"
HORIZON = int(os.environ.get("TRADING_HORIZON", "3"))
NUM_SCENARIOS = int(os.environ.get("TRADING_SCENARIOS", "512"))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    curves = {}
    summary: dict[str, object] = {
        "game": "trading",
        "evaluation": "Sampled root scenarios from the synthetic market library",
        "iterations": ITERATIONS,
        "eval_every": EVAL_EVERY,
        "eval_episodes": EVAL_EPISODES,
        "seeds": SEEDS,
        "horizon": HORIZON,
        "num_scenarios": NUM_SCENARIOS,
        "trainers": {},
    }
    builders = {
        "Vanilla CFR": lambda seed: MarketMakingGame(seed=seed, variant="vanilla", horizon=HORIZON, num_scenarios=NUM_SCENARIOS),
        "CFR+": lambda seed: MarketMakingGame(seed=seed, variant="cfr+", horizon=HORIZON, num_scenarios=NUM_SCENARIOS),
        "MCCFR": lambda seed: ExternalSamplingMCCFRTrainer(
            MarketMakingGame(seed=seed, variant="vanilla", horizon=HORIZON, num_scenarios=NUM_SCENARIOS),
            seed=seed,
        ),
        "Deep CFR": lambda seed: DeepCFRTrainer(
            MarketMakingGame(seed=seed, variant="vanilla", horizon=HORIZON, num_scenarios=NUM_SCENARIOS),
            seed=seed,
        ),
    }

    for label, builder in builders.items():
        aggregate, _, solvers = run_benchmark(
            builder=builder,
            seeds=SEEDS,
            iterations=ITERATIONS,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            eval_fn=exploitability,
        )
        curves[label] = aggregate
        print(f"{label}: final exploitability {aggregate['final_mean']:.4f} +/- {aggregate['final_std']:.4f}")
        policy_values = []
        for solver in solvers:
            game = solver.game if hasattr(solver, "game") else solver
            profile = solver.average_strategy_profile()
            policy_values.append(policy_value(game, profile, player=0, num_episodes=400, seed=game.seed + 99))
        summary["trainers"][label] = {
            "curve": aggregate,
            "final_exploitability_mean": aggregate["final_mean"],
            "final_exploitability_std": aggregate["final_std"],
            "self_play_value_mean": float(np.mean(policy_values)),
            "self_play_value_std": float(np.std(policy_values)),
        }

    plot_curves(
        curves=curves,
        title="Trading exploitability over training",
        output_path=RESULTS_DIR / "trading_exploitability.png",
    )

    representative_solver = MarketMakingGame(
        seed=0,
        variant="vanilla",
        horizon=HORIZON,
        num_scenarios=NUM_SCENARIOS,
    )
    representative_solver.train(
        iterations=ITERATIONS,
        eval_every=EVAL_EVERY,
        reset=True,
    )
    profile = representative_solver.average_strategy_profile()
    sample_strategy = profile.get(
        "round0_p0_inv0_moveflat_start",
        np.full(4, 0.25, dtype=float),
    )
    plot_strategy(
        actions=["narrow", "wide", "skewbid", "skewask"],
        strategy=sample_strategy,
        title="Trading average strategy at initial infoset",
        output_path=RESULTS_DIR / "trading_strategy.png",
    )
    summary["representative_strategy"] = {
        "infoset": "round0_p0_inv0_moveflat_start",
        "actions": ["narrow", "wide", "skewbid", "skewask"],
        "average_strategy": sample_strategy.tolist(),
    }

    save_summary(summary, SUMMARY_PATH)
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
