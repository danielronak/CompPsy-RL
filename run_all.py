"""
Master script: run all experiments in order.

Usage:
    python run_all.py                    # Run all experiments
    python run_all.py --exp 4.1          # Run only experiment 4.1
    python run_all.py --seeds 5          # Quick test with 5 seeds
    python run_all.py --exp 4.1 --seeds 5  # Quick test of experiment 4.1
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments.exp_dissonance import run_experiment_4_1
from src.experiments.exp_impostor_handicap import run_experiment_4_2, run_experiment_4_3
from src.experiments.exp_choice_overload import run_experiment_4_4


def main():
    parser = argparse.ArgumentParser(description="Run RL psychological biases experiments")
    parser.add_argument("--exp", type=str, default="all",
                        help="Which experiment to run: 4.1, 4.2, 4.3, 4.4, or 'all'")
    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of seeds per condition (default: 20)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    results = {}

    if args.exp in ("all", "4.1"):
        print("\n" + "=" * 70)
        print("PHASE 1: Experiment 4.1 -- Cognitive Dissonance")
        print("=" * 70)
        results["4.1"] = run_experiment_4_1(
            n_seeds=args.seeds,
            results_dir=args.results_dir,
        )

    if args.exp in ("all", "4.2"):
        print("\n" + "=" * 70)
        print("PHASE 2a: Experiment 4.2 -- Impostor Syndrome")
        print("=" * 70)
        results["4.2"] = run_experiment_4_2(
            n_seeds=args.seeds,
            results_dir=args.results_dir,
        )

    if args.exp in ("all", "4.3"):
        print("\n" + "=" * 70)
        print("PHASE 2b: Experiment 4.3 -- Self-Handicapping")
        print("=" * 70)
        results["4.3"] = run_experiment_4_3(
            n_seeds=args.seeds,
            results_dir=args.results_dir,
        )

    if args.exp in ("all", "4.4"):
        print("\n" + "=" * 70)
        print("PHASE 3: Experiment 4.4 -- Choice Overload")
        print("=" * 70)
        results["4.4"] = run_experiment_4_4(
            n_seeds=args.seeds,
            results_dir=args.results_dir,
        )

    if args.exp in ("all", "cross", "4"):
        print("\n" + "=" * 70)
        print("PHASE 4: Cross-Experiment Analysis & Generality")
        print("=" * 70)
        from src.analysis.shared_mechanism import analyze_shared_mechanism
        from src.analysis.human_benchmarks import generate_human_benchmarks_summary
        print("\nRunning Shared Mechanism Analysis...")
        analyze_shared_mechanism(results_dir=f"{args.results_dir}/cross_experiment_analysis")
        print("\nGenerating Human-Benchmark Summary Plots...")
        generate_human_benchmarks_summary(results_dir=f"{args.results_dir}/cross_experiment_analysis")

    print("\n" + "=" * 70)
    print("ALL REQUESTED EXPERIMENTS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
