"""
Visualization functions for SOC simulation results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_utility_curve(results_df: pd.DataFrame, 
                      save_path: Optional[str] = 'utility_gamma.png',
                      show: bool = False) -> None:
    """Plot utility vs gamma curve"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot utility curve
    ax.plot(results_df['gamma'], results_df['avg_utility'], 
            'b-', linewidth=2, marker='o', markersize=6)
    
    # Mark the optimal point
    optimal_idx = results_df['avg_utility'].idxmax()
    optimal_gamma = results_df.loc[optimal_idx, 'gamma']
    optimal_utility = results_df.loc[optimal_idx, 'avg_utility']
    
    ax.plot(optimal_gamma, optimal_utility, 'r*', markersize=15, 
            label=f'Optimal γ={optimal_gamma:.3f}')
    
    # Formatting
    ax.set_xlabel('Reward Shaping Coefficient (γ)', fontsize=12)
    ax.set_ylabel('Average Utility per Incident', fontsize=12)
    ax.set_title('Utility vs Gamma Coefficient', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotation for key points
    ax.annotate(f'Max utility: {optimal_utility:.2f}',
                xy=(optimal_gamma, optimal_utility),
                xytext=(optimal_gamma + 0.02, optimal_utility - 2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


def plot_caught_missed_crossover(results_df: pd.DataFrame,
                                save_path: Optional[str] = 'caught_missed_gamma.png',
                                show: bool = False) -> None:
    """Plot caught vs missed attacks crossover"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot caught and missed
    ax.plot(results_df['gamma'], results_df['caught_attacks'], 
            'g-', linewidth=2, marker='o', markersize=6, label='Caught')
    ax.plot(results_df['gamma'], results_df['missed_attacks'], 
            'r-', linewidth=2, marker='s', markersize=6, label='Missed')
    
    # Find crossover point (approximately)
    diff = np.abs(results_df['caught_attacks'].values - results_df['missed_attacks'].values)
    crossover_idx = np.argmin(diff)
    crossover_gamma = results_df.loc[crossover_idx, 'gamma']
    crossover_value = (results_df.loc[crossover_idx, 'caught_attacks'] + 
                      results_df.loc[crossover_idx, 'missed_attacks']) / 2
    
    # Mark crossover
    ax.plot(crossover_gamma, crossover_value, 'ko', markersize=10)
    ax.annotate(f'Crossover\nγ≈{crossover_gamma:.3f}',
                xy=(crossover_gamma, crossover_value),
                xytext=(crossover_gamma - 0.02, crossover_value + 50),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Reward Shaping Coefficient (γ)', fontsize=12)
    ax.set_ylabel('Number of Attacks', fontsize=12)
    ax.set_title('Caught vs Missed Attacks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


def create_results_table(results_df: pd.DataFrame,
                        save_path: Optional[str] = 'results_table.txt') -> str:
    """Create a formatted results table similar to the paper"""
    # Select key gamma values
    key_gammas = [0.00, 0.05, 0.09, 0.10]
    table_df = results_df[results_df['gamma'].isin(key_gammas)].copy()
    
    # Calculate additional metrics
    table_df['Min/Inc'] = table_df['total_time'] / table_df['total_incidents']
    table_df['Inv-rate'] = table_df['investigation_rate'] * 100
    
    # Format the table
    table_str = "γ\tUtility ↓\tMin/Inc\tInv-rate\tCaught\tMissed\n"
    table_str += "-" * 60 + "\n"
    
    for _, row in table_df.iterrows():
        gamma = row['gamma']
        utility = row['avg_utility']
        min_inc = row['Min/Inc']
        inv_rate = row['Inv-rate']
        caught = int(row['caught_attacks'])
        missed = int(row['missed_attacks'])
        
        # Mark optimal row
        prefix = "**" if gamma == 0.09 else "  "
        
        table_str += f"{prefix}{gamma:.2f}\t{utility:.1f}\t{min_inc:.1f}\t{inv_rate:.0f}%\t{caught}\t{missed}\n"
    
    print("\nResults Table:")
    print(table_str)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
            
    return table_str


def plot_simulation_metrics(metrics_history: List[Dict[str, float]],
                           save_path: Optional[str] = 'simulation_metrics.png',
                           show: bool = False) -> None:
    """Plot various metrics over simulation batches"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    batches = list(range(len(metrics_history)))
    
    # Utility over time
    utilities = [m['utility'] for m in metrics_history]
    axes[0, 0].plot(batches, utilities, 'b-')
    axes[0, 0].set_title('Batch Utility')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Utility')
    
    # Dismiss rate over time
    dismiss_rates = [m['dismissed'] / 1000 for m in metrics_history]  # Assuming batch size 1000
    axes[0, 1].plot(batches, dismiss_rates, 'g-')
    axes[0, 1].set_title('Dismiss Rate')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Dismiss Rate')
    axes[0, 1].set_ylim([0, 1])
    
    # Caught vs Missed
    caught = [m['caught'] for m in metrics_history]
    missed = [m['missed'] for m in metrics_history]
    axes[1, 0].plot(batches, caught, 'g-', label='Caught')
    axes[1, 0].plot(batches, missed, 'r-', label='Missed')
    axes[1, 0].set_title('Attack Detection')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # Investigation rate
    inv_rates = [m['investigated'] / 1000 for m in metrics_history]
    axes[1, 1].plot(batches, inv_rates, 'orange')
    axes[1, 1].set_title('Investigation Rate')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Investigation Rate')
    axes[1, 1].set_ylim([0, 1])
    
    plt.suptitle('Simulation Metrics Over Time', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_distribution(losses: List[float],
                          save_path: Optional[str] = 'loss_distribution.png',
                          show: bool = False) -> None:
    """Plot the distribution of losses from missed attacks"""
    if not losses:
        print("No losses to plot")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(losses, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax1.set_xlabel('Loss Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Losses from Missed Attacks')
    ax1.set_yscale('log')  # Log scale to see tail better
    
    # Log-log plot
    sorted_losses = np.sort(losses)[::-1]
    ranks = np.arange(1, len(sorted_losses) + 1)
    
    ax2.loglog(ranks, sorted_losses, 'b-', linewidth=2)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Loss Distribution (Log-Log Plot)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


def create_full_report(results_df: pd.DataFrame, 
                      annealing_result: Optional[tuple] = None,
                      output_dir: str = 'results/') -> None:
    """Create all visualizations and save report"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all plots
    plot_utility_curve(results_df, save_path=f'{output_dir}/utility_gamma.png')
    plot_caught_missed_crossover(results_df, save_path=f'{output_dir}/caught_missed_gamma.png')
    create_results_table(results_df, save_path=f'{output_dir}/results_table.txt')
    
    # Create summary report
    report = "# SOC Simulation Results Report\n\n"
    report += "## Summary\n"
    
    # Find optimal gamma from sweep
    optimal_idx = results_df['avg_utility'].idxmax()
    optimal_gamma = results_df.loc[optimal_idx, 'gamma']
    optimal_utility = results_df.loc[optimal_idx, 'avg_utility']
    
    report += f"- Optimal γ from sweep: {optimal_gamma:.3f}\n"
    report += f"- Maximum average utility: {optimal_utility:.2f}\n"
    
    if annealing_result:
        report += f"\n- Optimal γ from annealing: {annealing_result[0]:.3f}\n"
        report += f"- Annealing utility: {annealing_result[1]:.2f}\n"
    
    # Key findings
    report += "\n## Key Findings\n"
    
    # Compare baseline (γ=0) with optimal
    baseline = results_df[results_df['gamma'] == 0.0].iloc[0]
    optimal = results_df.loc[optimal_idx]
    
    utility_improvement = (optimal['avg_utility'] - baseline['avg_utility']) / abs(baseline['avg_utility'])
    miss_reduction = (baseline['missed_attacks'] - optimal['missed_attacks']) / baseline['missed_attacks']
    
    report += f"- Utility improvement: {utility_improvement:.1%}\n"
    report += f"- Miss reduction: {miss_reduction:.1%}\n"
    report += f"- Missed attacks reduced from {int(baseline['missed_attacks'])} to {int(optimal['missed_attacks'])}\n"
    
    # Save report
    with open(f'{output_dir}/report.md', 'w') as f:
        f.write(report)
        
    print(f"\nReport saved to {output_dir}/")
    print(report)