from __future__ import annotations

from pathlib import Path

import numpy as np

from deep_cfr import DeepCFRTrainer
from evaluate import (
    exact_exploitability,
    exact_policy_value,
    exploitability,
    plot_bars,
    plot_curves,
    run_benchmark,
    save_summary,
)
from mccfr import ExternalSamplingMCCFRTrainer
from poker_cfr import PokerGame
from trading_sim import MarketMakingGame


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = RESULTS_DIR / "analysis_summary.json"


def _trainer_game(solver: object) -> object:
    return solver.game if hasattr(solver, "game") else solver


def run() -> dict[str, object]:
    RESULTS_DIR.mkdir(exist_ok=True)
    seeds = [0, 1, 2]
    summary: dict[str, object] = {
        "seeds": seeds,
        "baseline_comparisons": {},
        "ablations": {},
    }

    poker_builders = {
        "Vanilla CFR": lambda seed: PokerGame(seed=seed, variant="vanilla"),
        "CFR+": lambda seed: PokerGame(seed=seed, variant="cfr+"),
        "MCCFR": lambda seed: ExternalSamplingMCCFRTrainer(PokerGame(seed=seed, variant="vanilla"), seed=seed),
        "Deep CFR": lambda seed: DeepCFRTrainer(PokerGame(seed=seed, variant="vanilla"), seed=seed),
    }
    poker_curves = {}
    poker_baselines = {}
    for label, builder in poker_builders.items():
        print(f"Running poker baseline: {label}")
        aggregate, _, solvers = run_benchmark(
            builder=builder,
            seeds=seeds,
            iterations=250 if label != "Deep CFR" else 200,
            eval_every=50,
            eval_episodes=300,
            eval_fn=exact_exploitability,
        )
        poker_curves[label] = aggregate
        self_play_values = []
        for solver in solvers:
            game = _trainer_game(solver)
            self_play_values.append(exact_policy_value(game, solver.average_strategy_profile(), player=0))
        poker_baselines[label] = {
            "curve": aggregate,
            "self_play_value_mean": float(np.mean(self_play_values)),
            "self_play_value_std": float(np.std(self_play_values)),
        }
    plot_curves(
        curves=poker_curves,
        title="Poker baseline comparison: exact exploitability",
        output_path=RESULTS_DIR / "analysis_poker_baselines.png",
    )
    summary["baseline_comparisons"]["poker"] = poker_baselines

    trading_builders = {
        "Vanilla CFR": lambda seed: MarketMakingGame(seed=seed, variant="vanilla", horizon=2, num_scenarios=128),
        "CFR+": lambda seed: MarketMakingGame(seed=seed, variant="cfr+", horizon=2, num_scenarios=128),
        "MCCFR": lambda seed: ExternalSamplingMCCFRTrainer(
            MarketMakingGame(seed=seed, variant="vanilla", horizon=2, num_scenarios=128),
            seed=seed,
        ),
        "Deep CFR": lambda seed: DeepCFRTrainer(
            MarketMakingGame(seed=seed, variant="vanilla", horizon=2, num_scenarios=128),
            seed=seed,
        ),
    }
    trading_curves = {}
    trading_baselines = {}
    for label, builder in trading_builders.items():
        print(f"Running trading baseline: {label}")
        aggregate, _, _ = run_benchmark(
            builder=builder,
            seeds=seeds,
            iterations=120 if label != "Deep CFR" else 100,
            eval_every=20,
            eval_episodes=120,
            eval_fn=exploitability,
        )
        trading_curves[label] = aggregate
        trading_baselines[label] = {"curve": aggregate}
    plot_curves(
        curves=trading_curves,
        title="Trading baseline comparison: sampled exploitability",
        output_path=RESULTS_DIR / "analysis_trading_baselines.png",
    )
    summary["baseline_comparisons"]["trading"] = trading_baselines

    # Ablation 1: sampled vs exact evaluation on poker.
    eval_ablation = {}
    for label, builder in {
        "Vanilla CFR": lambda seed: PokerGame(seed=seed, variant="vanilla"),
        "MCCFR": lambda seed: ExternalSamplingMCCFRTrainer(PokerGame(seed=seed, variant="vanilla"), seed=seed),
    }.items():
        print(f"Running poker evaluation ablation: {label}")
        exact_aggregate, _, solvers = run_benchmark(
            builder=builder,
            seeds=seeds,
            iterations=200,
            eval_every=200,
            eval_episodes=250,
            eval_fn=exact_exploitability,
        )
        exact_values = [exact_aggregate["final_mean"]]
        sampled_values = []
        for seed, solver in zip(seeds, solvers):
            game = _trainer_game(solver)
            sampled_values.append(exploitability(game, solver.average_strategy_profile(), num_episodes=300, seed=seed + 500))
        eval_ablation[label] = {
            "exact_exploitability": exact_aggregate["final_mean"],
            "sampled_exploitability_mean": float(np.mean(sampled_values)),
            "sampled_exploitability_std": float(np.std(sampled_values)),
        }
    plot_bars(
        values={f"{label} exact": values["exact_exploitability"] for label, values in eval_ablation.items()}
        | {f"{label} sampled": values["sampled_exploitability_mean"] for label, values in eval_ablation.items()},
        title="Poker ablation: exact vs sampled exploitability",
        ylabel="Exploitability",
        output_path=RESULTS_DIR / "ablation_poker_exact_vs_sampled.png",
    )
    summary["ablations"]["poker_exact_vs_sampled"] = eval_ablation

    # Ablation 2: Deep CFR hidden size on poker.
    hidden_ablation = {}
    for hidden_size in (64, 128, 256):
        print(f"Running Deep CFR hidden-size ablation: {hidden_size}")
        aggregate, _, _ = run_benchmark(
            builder=lambda seed, hs=hidden_size: DeepCFRTrainer(PokerGame(seed=seed, variant="vanilla"), seed=seed, hidden_size=hs),
            seeds=seeds,
            iterations=180,
            eval_every=45,
            eval_episodes=250,
            eval_fn=exact_exploitability,
        )
        hidden_ablation[f"hidden_{hidden_size}"] = aggregate
    plot_bars(
        values={label: values["final_mean"] for label, values in hidden_ablation.items()},
        title="Poker ablation: Deep CFR hidden size",
        ylabel="Final exact exploitability",
        output_path=RESULTS_DIR / "ablation_poker_hidden_size.png",
    )
    summary["ablations"]["poker_deep_hidden_size"] = hidden_ablation

    # Ablation 3: trading inventory penalty strength.
    sigma_ablation = {}
    for sigma in (0.0, 0.35, 0.7):
        print(f"Running trading sigma ablation: {sigma}")
        aggregate, _, _ = run_benchmark(
            builder=lambda seed, value=sigma: MarketMakingGame(
                seed=seed,
                variant="vanilla",
                horizon=2,
                sigma=value,
                num_scenarios=128,
            ),
            seeds=seeds,
            iterations=120,
            eval_every=30,
            eval_episodes=120,
            eval_fn=exploitability,
        )
        sigma_ablation[f"sigma_{sigma:.2f}"] = aggregate
    plot_bars(
        values={label: values["final_mean"] for label, values in sigma_ablation.items()},
        title="Trading ablation: inventory penalty strength",
        ylabel="Final sampled exploitability",
        output_path=RESULTS_DIR / "ablation_trading_sigma.png",
    )
    summary["ablations"]["trading_inventory_penalty"] = sigma_ablation

    # Ablation 4: trading horizon.
    horizon_ablation = {}
    for horizon in (2, 3):
        print(f"Running trading horizon ablation: {horizon}")
        aggregate, _, _ = run_benchmark(
            builder=lambda seed, value=horizon: MarketMakingGame(
                seed=seed,
                variant="vanilla",
                horizon=value,
                sigma=0.35,
                num_scenarios=128,
            ),
            seeds=seeds,
            iterations=70 if horizon == 2 else 50,
            eval_every=10 if horizon == 2 else 10,
            eval_episodes=80,
            eval_fn=exploitability,
        )
        horizon_ablation[f"horizon_{horizon}"] = aggregate
    plot_bars(
        values={label: values["final_mean"] for label, values in horizon_ablation.items()},
        title="Trading ablation: horizon length",
        ylabel="Final sampled exploitability",
        output_path=RESULTS_DIR / "ablation_trading_horizon.png",
    )
    summary["ablations"]["trading_horizon"] = horizon_ablation

    save_summary(summary, SUMMARY_PATH)
    return summary


if __name__ == "__main__":
    results = run()
    print(f"Saved analysis summary to {SUMMARY_PATH}")
    print(results.keys())
