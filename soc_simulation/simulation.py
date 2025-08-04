"""
Monte Carlo simulation runner for SOC experiments
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm

from .models import (
    Parameters, IncidentState, DecisionOutcome, 
    BayesianDecisionModel, DynamicEnvironment
)


@dataclass
class SimulationMetrics:
    """Metrics tracked during simulation"""
    total_utility: float = 0.0
    total_time: float = 0.0
    total_incidents: int = 0
    caught_attacks: int = 0
    missed_attacks: int = 0
    dismissed_incidents: int = 0
    investigated_incidents: int = 0
    
    # Detailed tracking
    utilities: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        if self.total_incidents == 0:
            return {}
            
        return {
            'avg_utility': self.total_utility / self.total_incidents,
            'avg_time_per_incident': self.total_time / self.total_incidents,
            'investigation_rate': self.investigated_incidents / self.total_incidents,
            'dismiss_rate': self.dismissed_incidents / self.total_incidents,
            'caught_attacks': self.caught_attacks,
            'missed_attacks': self.missed_attacks,
            'total_utility': self.total_utility,
            'total_time': self.total_time,
            'total_incidents': self.total_incidents
        }


class MonteCarloSimulation:
    """Runs Monte Carlo simulation of SOC operations"""
    
    def __init__(self, params: Parameters):
        self.params = params
        self.model = BayesianDecisionModel(params)
        self.environment = DynamicEnvironment(params)
        self.metrics = SimulationMetrics()
        
    def reset(self):
        """Reset simulation state"""
        self.environment = DynamicEnvironment(self.params)
        self.metrics = SimulationMetrics()
        
    def run_incident(self) -> Tuple[DecisionOutcome, float]:
        """Process a single incident"""
        # Generate incident
        incident = self.environment.generate_incident()
        
        # Calculate posterior
        incident.posterior = self.model.calculate_posterior(
            self.environment.current_prior, 
            incident.evidence_level
        )
        
        # Get queue factor
        queue_factor = self.environment.get_queue_factor()
        
        # Make decision
        outcome = self.model.make_triage_decision(incident, queue_factor)
        
        # Calculate utility change
        if outcome.missed:
            # Sample loss from heavy-tail distribution
            loss = self.model.sample_loss_miss()
            utility_change = -loss - outcome.time_cost  # Loss PLUS time cost
        else:
            utility_change = -outcome.time_cost  # Just time cost
            
        # Calculate reward for the agent (not used in decision, but tracked)
        utility_before = self.metrics.total_utility
        utility_after = utility_before + utility_change
        reward = self.model.calculate_reward(
            outcome, utility_before, utility_after
        )
        
        # Update queue
        if outcome.action in ['auto_investigate', 'manual_investigate']:
            self.environment.queue_size = min(
                self.environment.queue_size + 1, 
                self.params.queue_capacity
            )
        else:
            self.environment.queue_size = max(0, self.environment.queue_size - 0.1)
            
        outcome.utility_change = utility_change
        
        return outcome, reward
    
    def run_batch(self, batch_size: int) -> Dict[str, float]:
        """Run a batch of incidents"""
        batch_metrics = {
            'utility': 0,
            'time': 0,
            'caught': 0,
            'missed': 0,
            'dismissed': 0,
            'investigated': 0
        }
        
        for _ in range(batch_size):
            outcome, reward = self.run_incident()
            
            # Update batch metrics
            batch_metrics['utility'] += outcome.utility_change
            batch_metrics['time'] += outcome.time_cost
            
            if outcome.caught:
                batch_metrics['caught'] += 1
            if outcome.missed:
                batch_metrics['missed'] += 1
                
            if outcome.action == 'skip':
                batch_metrics['dismissed'] += 1
            elif outcome.action in ['auto_investigate', 'manual_investigate']:
                batch_metrics['investigated'] += 1
                
            # Update global metrics
            self.metrics.total_utility += outcome.utility_change
            self.metrics.total_time += outcome.time_cost
            self.metrics.total_incidents += 1
            
            if outcome.caught:
                self.metrics.caught_attacks += 1
            if outcome.missed:
                self.metrics.missed_attacks += 1
                self.metrics.losses.append(-outcome.utility_change)
                
            if outcome.action == 'skip':
                self.metrics.dismissed_incidents += 1
            elif outcome.action in ['auto_investigate', 'manual_investigate']:
                self.metrics.investigated_incidents += 1
                
            self.metrics.utilities.append(outcome.utility_change)
            self.metrics.times.append(outcome.time_cost)
            self.metrics.actions.append(outcome.action)
            
        # Update environment based on batch results
        dismiss_rate = batch_metrics['dismissed'] / batch_size
        self.environment.update_prior(dismiss_rate)
        self.environment.update_adversary(dismiss_rate)
        
        return batch_metrics
    
    def run_simulation(self, total_incidents: int = None, 
                      batch_size: int = None,
                      progress_bar: bool = True) -> SimulationMetrics:
        """Run full simulation"""
        if total_incidents is None:
            total_incidents = self.params.total_incidents
        if batch_size is None:
            batch_size = self.params.batch_size
            
        num_batches = total_incidents // batch_size
        
        if progress_bar:
            pbar = tqdm(total=num_batches, desc=f"γ={self.params.gamma:.2f}")
        
        for i in range(num_batches):
            self.run_batch(batch_size)
            
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix({
                    'utility': f"{self.metrics.total_utility:.1f}",
                    'missed': self.metrics.missed_attacks
                })
                
        if progress_bar:
            pbar.close()
            
        return self.metrics


def run_gamma_sweep(gamma_values: List[float], 
                   base_params: Parameters = None,
                   **kwargs) -> pd.DataFrame:
    """Run simulation for multiple gamma values"""
    if base_params is None:
        base_params = Parameters()
        
    results = []
    
    for gamma in gamma_values:
        # Create new parameters with this gamma
        params = Parameters(**{**base_params.__dict__, 'gamma': gamma})
        
        # Run simulation
        sim = MonteCarloSimulation(params)
        metrics = sim.run_simulation(**kwargs)
        
        # Get summary
        summary = metrics.get_summary()
        summary['gamma'] = gamma
        
        results.append(summary)
        
    return pd.DataFrame(results)


class SimulatedAnnealing:
    """Simulated annealing optimization for finding optimal gamma"""
    
    def __init__(self, base_params: Parameters = None):
        self.base_params = base_params or Parameters()
        self.best_gamma = 0.05
        self.best_utility = float('-inf')
        self.temperature = 1.0
        self.cooling_rate = 0.95
        
    def objective(self, gamma: float) -> float:
        """Objective function (negative utility to minimize)"""
        params = Parameters(**{**self.base_params.__dict__, 'gamma': gamma})
        sim = MonteCarloSimulation(params)
        metrics = sim.run_simulation(
            total_incidents=10000,  # Smaller sample for optimization
            progress_bar=False
        )
        return metrics.total_utility / metrics.total_incidents
    
    def neighbor(self, gamma: float) -> float:
        """Generate neighbor solution"""
        delta = np.random.normal(0, 0.01 * self.temperature)
        new_gamma = np.clip(gamma + delta, 0.0, 0.15)
        return new_gamma
    
    def accept_probability(self, current_util: float, new_util: float) -> float:
        """Probability of accepting worse solution"""
        if new_util > current_util:
            return 1.0
        return np.exp((new_util - current_util) / self.temperature)
    
    def optimize(self, max_iterations: int = 100) -> Tuple[float, float]:
        """Run simulated annealing optimization"""
        current_gamma = 0.05
        current_utility = self.objective(current_gamma)
        
        for i in tqdm(range(max_iterations), desc="Annealing"):
            # Generate neighbor
            new_gamma = self.neighbor(current_gamma)
            new_utility = self.objective(new_gamma)
            
            # Accept or reject
            if np.random.random() < self.accept_probability(current_utility, new_utility):
                current_gamma = new_gamma
                current_utility = new_utility
                
                # Update best
                if new_utility > self.best_utility:
                    self.best_gamma = new_gamma
                    self.best_utility = new_utility
                    
            # Cool down
            self.temperature *= self.cooling_rate
            
        return self.best_gamma, self.best_utility