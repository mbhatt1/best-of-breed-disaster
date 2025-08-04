#!/usr/bin/env python3
"""
Main script to run SOC simulation experiments and reproduce paper results
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import time

from soc_simulation.models import Parameters, OptimizationTarget
from soc_simulation.simulation import (
    MonteCarloSimulation,
    run_gamma_sweep,
    SimulatedAnnealing
)
from soc_simulation.visualization import (
    plot_utility_curve,
    plot_caught_missed_crossover,
    create_results_table,
    plot_simulation_metrics,
    plot_loss_distribution,
    create_full_report
)


def run_monte_carlo_sweep(output_dir: Path, params: Parameters = None) -> pd.DataFrame:
    """Run the Monte Carlo sweep experiment"""
    print("\n" + "="*60)
    print("Running Monte Carlo Sweep Experiment")
    print("="*60)
    
    # Define gamma values to test (as per paper)
    gamma_values = np.arange(0.0, 0.11, 0.01)
    
    # Run sweep
    start_time = time.time()
    results_df = run_gamma_sweep(
        gamma_values=gamma_values,
        base_params=params,
        total_incidents=30000,
        batch_size=1000,
        progress_bar=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nSweep completed in {elapsed:.1f} seconds")
    
    # Save results
    results_df.to_csv(output_dir / 'sweep_results.csv', index=False)
    
    # Print key results similar to paper's Table 1
    print("\nKey Results (similar to Table 1 in paper):")
    create_results_table(results_df)
    
    return results_df


def run_annealing_optimization(output_dir: Path, params: Parameters = None) -> tuple:
    """Run the simulated annealing optimization"""
    print("\n" + "="*60)
    print("Running Simulated Annealing Optimization")
    print("="*60)
    
    optimizer = SimulatedAnnealing(base_params=params)
    
    start_time = time.time()
    optimal_gamma, optimal_utility = optimizer.optimize(max_iterations=50)
    elapsed = time.time() - start_time
    
    print(f"\nOptimization completed in {elapsed:.1f} seconds")
    print(f"Optimal γ: {optimal_gamma:.4f}")
    print(f"Optimal utility: {optimal_utility:.2f}")
    
    # Save results
    annealing_results = {
        'optimal_gamma': optimal_gamma,
        'optimal_utility': optimal_utility,
        'optimization_time': elapsed
    }
    
    with open(output_dir / 'annealing_results.json', 'w') as f:
        json.dump(annealing_results, f, indent=2)
    
    return optimal_gamma, optimal_utility


def run_detailed_simulation(gamma: float, output_dir: Path, 
                          params: Parameters = None) -> None:
    """Run a detailed simulation for a specific gamma value"""
    print(f"\n" + "="*60)
    print(f"Running Detailed Simulation for γ={gamma:.3f}")
    print("="*60)
    
    # Create parameters with specified gamma
    if params is None:
        params = Parameters()
    params.gamma = gamma
    
    # Create simulation
    sim = MonteCarloSimulation(params)
    
    # Track metrics per batch
    batch_metrics = []
    
    # Run simulation batch by batch
    num_batches = params.total_incidents // params.batch_size
    
    for i in range(num_batches):
        metrics = sim.run_batch(params.batch_size)
        batch_metrics.append(metrics)
        
        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}/{num_batches} - "
                  f"Utility: {metrics['utility']:.1f}, "
                  f"Caught: {metrics['caught']}, "
                  f"Missed: {metrics['missed']}")
    
    # Get final metrics
    final_metrics = sim.metrics.get_summary()
    
    print(f"\nFinal Results:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Plot batch metrics
    plot_simulation_metrics(batch_metrics, 
                          save_path=output_dir / f'batch_metrics_gamma_{gamma:.3f}.png')
    
    # Plot loss distribution if there were misses
    if sim.metrics.losses:
        plot_loss_distribution(sim.metrics.losses,
                             save_path=output_dir / f'loss_distribution_gamma_{gamma:.3f}.png')


def validate_results(sweep_df: pd.DataFrame, paper_values: dict) -> None:
    """Validate simulation results against paper values"""
    print("\n" + "="*60)
    print("Validating Results Against Paper")
    print("="*60)
    
    for gamma, expected in paper_values.items():
        # Find closest gamma in results
        idx = np.argmin(np.abs(sweep_df['gamma'] - gamma))
        actual = sweep_df.iloc[idx]
        
        print(f"\nγ = {gamma}:")
        print(f"  Utility - Expected: {expected['utility']}, "
              f"Actual: {actual['avg_utility']:.1f}")
        print(f"  Caught - Expected: {expected['caught']}, "
              f"Actual: {int(actual['caught_attacks'])}")
        print(f"  Missed - Expected: {expected['missed']}, "
              f"Actual: {int(actual['missed_attacks'])}")


def run_optimization_comparison(output_dir: Path, params: Parameters = None) -> pd.DataFrame:
    """Run comparison across all optimization targets"""
    print("\n" + "="*60)
    print("Running Optimization Target Comparison")
    print("="*60)
    
    if params is None:
        params = Parameters()
    
    all_results = []
    
    # Test each optimization target
    for target in OptimizationTarget:
        print(f"\nTesting optimization target: {target.value}")
        
        # Set optimization target
        test_params = Parameters(**{**params.__dict__, 'optimization_target': target})
        
        # Run simulation with different gamma values
        gamma_values = [0.0, 0.03, 0.06, 0.09] if target == OptimizationTarget.BALANCED else [0.05]
        
        for gamma in gamma_values:
            test_params.gamma = gamma
            sim = MonteCarloSimulation(test_params)
            metrics = sim.run_simulation(
                total_incidents=params.total_incidents,  # Use full simulation size
                batch_size=params.batch_size,
                progress_bar=True
            )
            
            summary = metrics.get_summary()
            summary['optimization_target'] = target.value
            summary['gamma'] = gamma
            
            all_results.append(summary)
            
            print(f"  γ={gamma:.2f}: Utility={summary['avg_utility']:.1f}, "
                  f"Caught={summary['caught_attacks']}, "
                  f"Missed={summary['missed_attacks']}")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(output_dir / 'optimization_comparison.csv', index=False)
    
    # Create summary table
    print("\n" + "="*60)
    print("Optimization Target Comparison Summary")
    print("="*60)
    print(f"{'Target':<20} {'Utility':<10} {'Caught':<10} {'Missed':<10} {'Time/Inc':<10}")
    print("-" * 60)
    
    for target in OptimizationTarget:
        rows = comparison_df[comparison_df['optimization_target'] == target.value]
        if not rows.empty:
            best_row = rows.loc[rows['avg_utility'].idxmax()]
            print(f"{target.value:<20} {best_row['avg_utility']:<10.1f} "
                  f"{int(best_row['caught_attacks']):<10} "
                  f"{int(best_row['missed_attacks']):<10} "
                  f"{best_row['avg_time_per_incident']:<10.1f}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Run SOC simulation experiments')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--sweep-only', action='store_true',
                       help='Run only the gamma sweep experiment')
    parser.add_argument('--annealing-only', action='store_true',
                       help='Run only the annealing optimization')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Run detailed simulation for specific gamma')
    parser.add_argument('--validate', action='store_true',
                       help='Validate results against paper values')
    parser.add_argument('--optimize-for', type=str,
                       choices=[t.value for t in OptimizationTarget],
                       default='balanced',
                       help='Optimization target to use')
    parser.add_argument('--all', action='store_true',
                       help='Run comparison across all optimization targets')
    parser.add_argument('--incidents', type=int, default=30000,
                       help='Number of incidents to simulate (default: 30000)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize parameters (matching paper values)
    params = Parameters(
        prior=0.02,
        c_skip=0.5,
        c_triage=1.0,
        c_auto=5.0,
        c_manual=30.0,
        loss_mean=6.0,
        loss_std=1.2,
        total_incidents=args.incidents,
        batch_size=1000,
        optimization_target=OptimizationTarget(args.optimize_for)
    )
    
    # Save parameters (convert enum to string for JSON serialization)
    params_dict = params.__dict__.copy()
    params_dict['optimization_target'] = params_dict['optimization_target'].value
    with open(output_dir / 'parameters.json', 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    results = {}
    
    # Run experiments based on arguments
    if args.all:
        # Run comparison across all optimization targets
        comparison_df = run_optimization_comparison(output_dir, params)
        results['comparison'] = comparison_df
        
        # Create visualization comparing all targets
        create_optimization_comparison_plot(comparison_df, output_dir)
        
    elif args.gamma is not None:
        # Run detailed simulation for specific gamma
        run_detailed_simulation(args.gamma, output_dir, params)
        
    elif args.annealing_only:
        # Run only annealing
        annealing_result = run_annealing_optimization(output_dir, params)
        results['annealing'] = annealing_result
        
    elif args.sweep_only:
        # Run only sweep
        sweep_df = run_monte_carlo_sweep(output_dir, params)
        results['sweep'] = sweep_df
        
    else:
        # Run all experiments
        print("\nRunning Complete Experiment Suite")
        
        # 1. Monte Carlo sweep
        sweep_df = run_monte_carlo_sweep(output_dir, params)
        results['sweep'] = sweep_df
        
        # 2. Simulated annealing
        annealing_result = run_annealing_optimization(output_dir, params)
        results['annealing'] = annealing_result
        
        # 3. Generate all visualizations
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)
        
        plot_utility_curve(sweep_df, save_path=output_dir / 'utility_gamma.png')
        plot_caught_missed_crossover(sweep_df, 
                                   save_path=output_dir / 'caught_missed_gamma.png')
        
        # 4. Create full report
        create_full_report(sweep_df, annealing_result, str(output_dir))
        
        # 5. Run detailed simulation for optimal gamma
        optimal_idx = sweep_df['avg_utility'].idxmax()
        optimal_gamma = sweep_df.loc[optimal_idx, 'gamma']
        run_detailed_simulation(optimal_gamma, output_dir, params)
        
        # 6. Validate against paper if requested
        if args.validate:
            # Paper values from Table 1
            paper_values = {
                0.00: {'utility': -43.3, 'caught': 0, 'missed': 535},
                0.05: {'utility': -18.4, 'caught': 245, 'missed': 180},
                0.09: {'utility': -11.1, 'caught': 381, 'missed': 25},
                0.10: {'utility': -11.2, 'caught': 393, 'missed': 23}
            }
            validate_results(sweep_df, paper_values)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}/")
    print("="*60)


def create_optimization_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path):
    """Create visualization comparing all optimization targets"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Utility comparison
    ax = axes[0, 0]
    targets = comparison_df['optimization_target'].unique()
    utilities = []
    for target in targets:
        rows = comparison_df[comparison_df['optimization_target'] == target]
        if not rows.empty:
            utilities.append(rows['avg_utility'].max())
        else:
            utilities.append(0)
    
    ax.bar(targets, utilities)
    ax.set_xlabel('Optimization Target')
    ax.set_ylabel('Best Average Utility')
    ax.set_title('Utility by Optimization Target')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Caught vs Missed
    ax = axes[0, 1]
    caught = []
    missed = []
    for target in targets:
        rows = comparison_df[comparison_df['optimization_target'] == target]
        if not rows.empty:
            best_row = rows.loc[rows['avg_utility'].idxmax()]
            caught.append(best_row['caught_attacks'])
            missed.append(best_row['missed_attacks'])
        else:
            caught.append(0)
            missed.append(0)
    
    x = np.arange(len(targets))
    width = 0.35
    ax.bar(x - width/2, caught, width, label='Caught', color='green')
    ax.bar(x + width/2, missed, width, label='Missed', color='red')
    ax.set_xlabel('Optimization Target')
    ax.set_ylabel('Number of Attacks')
    ax.set_title('Detection Performance by Target')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45)
    ax.legend()
    
    # 3. Time efficiency
    ax = axes[1, 0]
    times = []
    for target in targets:
        rows = comparison_df[comparison_df['optimization_target'] == target]
        if not rows.empty:
            times.append(rows['avg_time_per_incident'].min())
        else:
            times.append(0)
    
    ax.bar(targets, times, color='orange')
    ax.set_xlabel('Optimization Target')
    ax.set_ylabel('Avg Time per Incident (min)')
    ax.set_title('Time Efficiency by Target')
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Investigation rate
    ax = axes[1, 1]
    inv_rates = []
    for target in targets:
        rows = comparison_df[comparison_df['optimization_target'] == target]
        if not rows.empty:
            inv_rates.append(rows['investigation_rate'].mean() * 100)
        else:
            inv_rates.append(0)
    
    ax.bar(targets, inv_rates, color='blue')
    ax.set_xlabel('Optimization Target')
    ax.set_ylabel('Investigation Rate (%)')
    ax.set_title('Investigation Rate by Target')
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Optimization Paradox: Comparison Across Targets', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to {output_dir}/optimization_comparison.png")


if __name__ == '__main__':
    main()