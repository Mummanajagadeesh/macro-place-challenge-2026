"""
Evaluation harness for macro placement competition.

This module provides the main evaluation pipeline that:
1. Loads benchmarks
2. Runs participant placers with timeout
3. Validates placements
4. Computes metrics (proxy cost)
5. Aggregates scores across benchmarks
6. Generates leaderboard
"""

import time
import signal
from typing import Type, Optional, Dict, Any, List
from pathlib import Path
import json
import torch

from marco_place.data.tensor_schema import CircuitTensorData
from marco_place.data.dataset import MacroPlacementDataset
from marco_place.metrics.proxy_cost import (
    compute_proxy_cost,
    compute_improvement,
    compute_aggregate_score,
    is_prize_eligible
)
from marco_place.validation.legality import PlacementValidator


class TimeoutException(Exception):
    """Exception raised when placer exceeds time limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Placer exceeded time limit")


class EvaluationHarness:
    """
    Main evaluation harness for competition submissions.

    Evaluates a placer on all benchmarks and computes aggregate scores.
    """

    def __init__(
        self,
        benchmark_dir: str = "benchmarks/processed",
        split: str = "public",
        output_dir: Optional[str] = None,
        timeout_seconds: int = 3600,  # 1 hour
        baseline_scores: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize EvaluationHarness.

        Args:
            benchmark_dir: Directory containing processed benchmarks
            split: Which benchmark split to use ('public' or 'hidden')
            output_dir: Directory to save results (default: results/)
            timeout_seconds: Maximum time per benchmark (default: 3600 = 1 hour)
            baseline_scores: Dictionary of baseline scores for comparison
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.split = split
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.timeout_seconds = timeout_seconds
        self.baseline_scores = baseline_scores or {}

        # Load dataset
        self.dataset = MacroPlacementDataset(benchmark_dir, split=split)

        print(f"Initialized EvaluationHarness:")
        print(f"  Benchmarks: {len(self.dataset)}")
        print(f"  Split: {split}")
        print(f"  Timeout: {timeout_seconds}s per benchmark")

    def evaluate_submission(
        self,
        placer_class: Type,
        submission_name: str = "submission",
        benchmark_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a submission on all benchmarks.

        Args:
            placer_class: Class with .place(circuit_data) -> placement method
            submission_name: Name for this submission (for results)
            benchmark_names: Optional list of specific benchmarks to evaluate
                           (default: all benchmarks)
            verbose: Print detailed progress

        Returns:
            Results dictionary with per-benchmark metrics and aggregate score
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"Evaluating Submission: {submission_name}")
            print("=" * 70)

        results = {
            'submission_name': submission_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': {},
            'summary': {}
        }

        # Get benchmarks to evaluate
        if benchmark_names:
            benchmarks = [(name, self.dataset.get_by_name(name)) for name in benchmark_names]
        else:
            benchmarks = [
                (self.dataset.get_benchmark_names()[i], self.dataset[i])
                for i in range(len(self.dataset))
            ]

        # Evaluate each benchmark
        benchmark_results = []
        for benchmark_name, circuit_data in benchmarks:
            if verbose:
                print(f"\n--- Evaluating {benchmark_name} ---")
                print(f"Design: {circuit_data}")

            result = self._evaluate_single_benchmark(
                placer_class,
                circuit_data,
                benchmark_name,
                verbose=verbose
            )

            results['benchmarks'][benchmark_name] = result
            benchmark_results.append((benchmark_name, result))

        # Compute aggregate statistics
        results['summary'] = self._compute_summary(benchmark_results, verbose=verbose)

        # Save results
        self._save_results(results, submission_name)

        if verbose:
            print("\n" + "=" * 70)
            print("Evaluation Complete")
            print("=" * 70)
            self._print_summary(results['summary'])

        return results

    def _evaluate_single_benchmark(
        self,
        placer_class: Type,
        circuit_data: CircuitTensorData,
        benchmark_name: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Evaluate placer on a single benchmark."""
        result = {
            'benchmark_name': benchmark_name,
            'num_macros': circuit_data.num_macros,
            'num_nets': circuit_data.num_nets,
        }

        try:
            # Instantiate placer
            placer = placer_class()

            # Run placement with timeout
            if verbose:
                print(f"Running placer (timeout: {self.timeout_seconds}s)...")

            start_time = time.time()

            # Set timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                result_placement = placer.place(circuit_data)
                runtime = time.time() - start_time

                # Cancel alarm
                signal.alarm(0)

                # Handle flexible return type
                # placer.place() can return either:
                #   - single tensor: macro_placement only
                #   - tuple: (macro_placement, cell_placement)
                if isinstance(result_placement, tuple):
                    macro_placement, cell_placement = result_placement
                    if verbose:
                        has_cells = cell_placement is not None
                        print(f"✓ Completed in {runtime:.2f}s (placed {'macros + cells' if has_cells else 'macros only'})")
                else:
                    macro_placement = result_placement
                    cell_placement = None
                    if verbose:
                        print(f"✓ Completed in {runtime:.2f}s (placed macros only)")

            except TimeoutException:
                if verbose:
                    print(f"✗ Timeout after {self.timeout_seconds}s")
                result['error'] = f'Timeout ({self.timeout_seconds}s)'
                result['valid'] = False
                return result

            result['runtime'] = runtime
            result['placed_cells'] = cell_placement is not None

            # Validate placement (macro placement is primary validation target)
            if verbose:
                print("Validating macro placement...")

            validator = PlacementValidator(circuit_data)
            is_valid, violations = validator.validate(macro_placement)

            result['valid'] = is_valid

            if not is_valid:
                result['violations'] = violations
                if verbose:
                    print(f"✗ Invalid placement:")
                    for violation in violations[:3]:  # Show first 3
                        print(f"  - {violation}")
                return result

            if verbose:
                print("✓ Placement is valid")

            # Compute metrics (on macro placement)
            if verbose:
                print("Computing metrics...")

            costs = compute_proxy_cost(macro_placement, circuit_data)
            result.update(costs)

            if verbose:
                print(f"  Proxy cost: {costs['proxy_cost']:.6f}")
                print(f"    - Wirelength: {costs['wirelength_cost']:.6f}")
                print(f"    - Density:    {costs['density_cost']:.6f}")
                print(f"    - Congestion: {costs['congestion_cost']:.6f}")

            # Compute improvement over baseline (if available)
            if benchmark_name in self.baseline_scores:
                baseline_cost = self.baseline_scores[benchmark_name]['proxy_cost']
                improvement = compute_improvement(costs['proxy_cost'], baseline_cost)
                result['baseline_cost'] = baseline_cost
                result['improvement_pct'] = improvement

                if verbose:
                    print(f"  Baseline cost: {baseline_cost:.6f}")
                    print(f"  Improvement: {improvement:+.2f}%")

        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            result['error'] = str(e)
            result['valid'] = False
            import traceback
            result['traceback'] = traceback.format_exc()

        return result

    def _compute_summary(
        self,
        benchmark_results: List[tuple],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Compute aggregate statistics across all benchmarks."""
        summary = {
            'total_benchmarks': len(benchmark_results),
            'valid_benchmarks': 0,
            'failed_benchmarks': 0,
            'total_runtime': 0.0,
        }

        valid_results = []
        improvements = []

        for benchmark_name, result in benchmark_results:
            if result.get('valid', False):
                summary['valid_benchmarks'] += 1
                valid_results.append((benchmark_name, result))

                if 'runtime' in result:
                    summary['total_runtime'] += result['runtime']

                if 'improvement_pct' in result:
                    improvements.append(result['improvement_pct'])
            else:
                summary['failed_benchmarks'] += 1

        # Compute aggregate score
        if improvements:
            summary['aggregate_score'] = compute_aggregate_score(improvements)
            summary['prize_eligible'] = is_prize_eligible(summary['aggregate_score'])
            summary['mean_improvement'] = sum(improvements) / len(improvements)
            summary['min_improvement'] = min(improvements)
            summary['max_improvement'] = max(improvements)
        else:
            summary['aggregate_score'] = 0.0
            summary['prize_eligible'] = False

        return summary

    def _save_results(self, results: Dict[str, Any], submission_name: str):
        """Save results to JSON file."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        output_file = self.output_dir / f"{submission_name}_results.json"

        # Convert tensors to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        results_json = convert_for_json(results)

        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary."""
        print(f"\nSummary:")
        print(f"  Total benchmarks:    {summary['total_benchmarks']}")
        print(f"  Valid:               {summary['valid_benchmarks']}")
        print(f"  Failed:              {summary['failed_benchmarks']}")
        print(f"  Total runtime:       {summary['total_runtime']:.2f}s")

        if summary.get('aggregate_score') is not None:
            print(f"\n  Aggregate Score:     {summary['aggregate_score']:+.2f}%")
            print(f"  Mean Improvement:    {summary.get('mean_improvement', 0):+.2f}%")
            print(f"  Min Improvement:     {summary.get('min_improvement', 0):+.2f}%")
            print(f"  Max Improvement:     {summary.get('max_improvement', 0):+.2f}%")

            if summary.get('prize_eligible', False):
                print(f"\n  🏆 PRIZE ELIGIBLE: This submission qualifies for the $20K prize!")
            else:
                print(f"\n  ✗ Not prize eligible (must beat Circuit Training baseline)")


def load_baseline_scores(baseline_file: str = "benchmarks/metadata/baseline_scores.json") -> Dict:
    """
    Load baseline scores from JSON file.

    Args:
        baseline_file: Path to baseline scores JSON

    Returns:
        Dictionary mapping benchmark names to baseline metrics
    """
    baseline_path = Path(baseline_file)

    if not baseline_path.exists():
        print(f"Warning: Baseline scores not found at {baseline_file}")
        return {}

    with open(baseline_path, 'r') as f:
        baselines = json.load(f)

    return baselines
