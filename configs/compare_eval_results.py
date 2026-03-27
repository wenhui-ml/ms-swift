#!/usr/bin/env python3
"""
汇总 lm-evaluation-harness 评测结果，生成对比表格。

Usage:
    python configs/compare_eval_results.py [eval_results_dir]
"""

import json
import sys
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """Load all evaluation results from a directory."""
    all_results = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        # lm_eval saves results as JSON files
        json_files = list(model_dir.rglob("results_*.json"))
        if not json_files:
            # Try the direct results.json format
            json_files = list(model_dir.rglob("*.json"))
            json_files = [f for f in json_files if f.name != "eval.log"]

        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "results" in data:
                    all_results[model_dir.name] = data["results"]
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    return all_results


def print_comparison(all_results: dict):
    """Print a formatted comparison table."""
    if not all_results:
        print("No results found.")
        return

    # Collect all tasks across all models
    all_tasks = set()
    for results in all_results.values():
        for task_name in results:
            all_tasks.add(task_name)

    # Focus on top-level aggregate tasks
    top_tasks = sorted([t for t in all_tasks if "," not in t and "_" not in t or t in [
        "mmlu", "arc_easy", "arc_challenge", "hellaswag", "winogrande",
        "truthfulqa_mc2", "gsm8k",
    ]])

    # If no top-level tasks found, use all tasks
    if not top_tasks:
        top_tasks = sorted(all_tasks)

    model_names = sorted(all_results.keys())

    # Print header
    print("\n" + "=" * 100)
    print("Model Comparison (acc / acc_norm where available)")
    print("=" * 100)

    # Column widths
    task_width = 20
    model_width = max(len(n) for n in model_names) + 2

    header = f"{'Task':<{task_width}}"
    for name in model_names:
        header += f" | {name:>{model_width}}"
    print(header)
    print("-" * len(header))

    # Print each task
    avg_scores = {name: [] for name in model_names}

    for task in top_tasks:
        row = f"{task:<{task_width}}"
        for name in model_names:
            results = all_results[name]
            if task in results:
                task_results = results[task]
                # Try different metric names
                score = None
                for metric in ["acc,none", "acc_norm,none", "acc", "acc_norm", "exact_match,none"]:
                    if metric in task_results:
                        score = task_results[metric]
                        break
                if score is not None:
                    row += f" | {score:>{model_width}.4f}"
                    avg_scores[name].append(score)
                else:
                    row += f" | {'N/A':>{model_width}}"
            else:
                row += f" | {'--':>{model_width}}"
        print(row)

    # Print average
    print("-" * len(header))
    row = f"{'AVERAGE':<{task_width}}"
    for name in model_names:
        scores = avg_scores[name]
        if scores:
            avg = sum(scores) / len(scores)
            row += f" | {avg:>{model_width}.4f}"
        else:
            row += f" | {'--':>{model_width}}"
    print(row)
    print("=" * 100)

    # Print random baseline for reference
    print("\nRandom baselines: MMLU=0.25, ARC=0.25, HellaSwag=0.25, WinoGrande=0.50, TruthfulQA≈0.50")
    print("Qwen3-0.6B (official, 5-shot MMLU) ≈ 0.47")


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_results")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run evaluations first: bash configs/eval_lm_harness.sh all")
        sys.exit(1)

    all_results = load_results(results_dir)
    print(f"\nFound results for {len(all_results)} models in {results_dir}/")

    print_comparison(all_results)


if __name__ == "__main__":
    main()
