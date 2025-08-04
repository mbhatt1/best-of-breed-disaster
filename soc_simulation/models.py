"""
Core models for SOC simulation
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy.stats import lognorm
from enum import Enum


class OptimizationTarget(Enum):
    """Different optimization targets that can create paradoxes"""
    SPEED = "speed"  # Original - minimize time
    ACCURACY = "accuracy"  # Only flag high-confidence threats
    FALSE_POSITIVE_RATE = "false_positive_rate"  # Minimize false alarms
    RESOURCE_EFFICIENCY = "resource_efficiency"  # Minimize compute/memory
    ALERT_VOLUME = "alert_volume"  # Minimize analyst workload
    COMPLIANCE = "compliance"  # Optimize for regulatory requirements
    USER_EXPERIENCE = "user_experience"  # Minimize user friction
    COST = "cost"  # Minimize monetary cost per incident
    COVERAGE = "coverage"  # Maximize detection coverage
    AUTOMATION = "automation"  # Maximize automated decisions
    BALANCED = "balanced"  # Original gamma-based approach


@dataclass
class Parameters:
    """Simulation parameters from the paper"""
    # Prior probability that an incident is malicious
    prior: float = 0.02
    
    # Time costs (in minutes)
    c_skip: float = 0.5
    c_triage: float = 1.0
    c_auto: float = 5.0
    c_manual: float = 30.0
    
    # Loss distribution parameters (LogNormal)
    loss_mean: float = 6.0
    loss_std: float = 1.2
    
    # Queue contention parameter
    alpha_queue: float = 0.5
    queue_capacity: int = 100
    
    # Adversary adaptation parameters
    adversary_decay: float = 0.97
    dismiss_threshold: float = 0.8
    
    # Alert evidence probabilities
    p_evidence_high_given_malicious: float = 0.8
    p_evidence_high_given_benign: float = 0.1
    
    # Reward shaping coefficient
    gamma: float = 0.05
    
    # Optimization target
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
    
    # Simulation settings
    batch_size: int = 1000
    total_incidents: int = 30000
    

@dataclass
class IncidentState:
    """State of a single incident"""
    is_malicious: bool
    evidence_level: str  # 'High', 'Medium', 'Low'
    posterior: float = 0.0
    
    
@dataclass
class DecisionOutcome:
    """Outcome of a triage decision"""
    action: str  # 'skip', 'triage', 'auto_investigate', 'manual_investigate'
    time_cost: float
    caught: bool
    missed: bool
    utility_change: float = 0.0


class BayesianDecisionModel:
    """Implements the two-stage Bayesian decision model"""
    
    def __init__(self, params: Parameters):
        self.params = params
        self._loss_dist = lognorm(s=params.loss_std, scale=np.exp(params.loss_mean))
        
    def calculate_posterior(self, prior: float, evidence: str) -> float:
        """Calculate posterior probability given evidence"""
        if evidence == 'High':
            p_e_given_h1 = self.params.p_evidence_high_given_malicious
            p_e_given_h0 = self.params.p_evidence_high_given_benign
        elif evidence == 'Medium':
            p_e_given_h1 = 0.5
            p_e_given_h0 = 0.3
        else:  # Low
            p_e_given_h1 = 0.2
            p_e_given_h0 = 0.6
            
        # Bayes' theorem
        p_e = p_e_given_h1 * prior + p_e_given_h0 * (1 - prior)
        posterior = (p_e_given_h1 * prior) / p_e if p_e > 0 else prior
        
        return posterior
    
    def global_posterior_threshold(self) -> float:
        """Calculate the global optimal posterior threshold"""
        expected_loss = self._loss_dist.mean()
        c_invest = self.params.c_auto  # Using auto-investigate as default
        c_skip = self.params.c_skip
        g_fp = 0.1  # False positive goodwill cost (approximation)
        
        threshold = (c_invest - c_skip + g_fp) / (expected_loss + g_fp)
        return threshold
    
    def expected_loss_miss(self) -> float:
        """Expected loss from missing a malicious incident"""
        return self._loss_dist.mean()
    
    def sample_loss_miss(self) -> float:
        """Sample a loss value from the log-normal distribution"""
        return self._loss_dist.rvs()
    
    def calculate_reward(self, decision_outcome: DecisionOutcome, 
                        prior_utility: float, post_utility: float) -> float:
        """Calculate agent reward with reward shaping"""
        time_cost = decision_outcome.time_cost
        utility_change = post_utility - prior_utility
        
        reward = -time_cost + self.params.gamma * utility_change
        return reward
    
    def make_triage_decision(self, incident: IncidentState,
                           queue_factor: float = 1.0) -> DecisionOutcome:
        """Make a triage decision based on current parameters and queue state"""
        # Calculate adjusted costs based on queue contention
        adjusted_auto_cost = self.params.c_auto * queue_factor
        adjusted_manual_cost = self.params.c_manual * queue_factor
        
        # Get threshold based on optimization target
        threshold = self._get_decision_threshold(incident, queue_factor)
        
        # Make decision based on posterior vs threshold
        if incident.posterior < threshold:
            # Skip - don't investigate
            return DecisionOutcome(
                action='skip',
                time_cost=self.params.c_skip,
                caught=False,
                missed=incident.is_malicious
            )
        else:
            # Investigate - choose method based on optimization target
            if self.params.optimization_target == OptimizationTarget.COST:
                # Always use cheapest investigation method
                caught = incident.is_malicious
                return DecisionOutcome(
                    action='auto_investigate',
                    time_cost=self.params.c_triage + adjusted_auto_cost,
                    caught=caught,
                    missed=False
                )
            elif self.params.optimization_target == OptimizationTarget.ACCURACY:
                # Use manual investigation for uncertain cases
                if 0.3 < incident.posterior < 0.7:
                    caught = incident.is_malicious
                    return DecisionOutcome(
                        action='manual_investigate',
                        time_cost=self.params.c_triage + adjusted_manual_cost,
                        caught=caught,
                        missed=False
                    )
            
            # Default: auto-investigate
            caught = incident.is_malicious
            return DecisionOutcome(
                action='auto_investigate',
                time_cost=self.params.c_triage + adjusted_auto_cost,
                caught=caught,
                missed=False
            )
    
    def _get_decision_threshold(self, incident: IncidentState, queue_factor: float) -> float:
        """Get decision threshold based on optimization target"""
        base_threshold = self.global_posterior_threshold()
        
        if self.params.optimization_target == OptimizationTarget.SPEED:
            # Never investigate - pure speed optimization
            return 1.0
            
        elif self.params.optimization_target == OptimizationTarget.ACCURACY:
            # Only investigate very high confidence threats
            return 0.95 - (0.5 * self.params.gamma)
            
        elif self.params.optimization_target == OptimizationTarget.FALSE_POSITIVE_RATE:
            # Very high threshold to minimize false positives
            return 0.85 - (0.3 * self.params.gamma)
            
        elif self.params.optimization_target == OptimizationTarget.RESOURCE_EFFICIENCY:
            # Adjust threshold based on queue load
            if queue_factor > 1.5:
                return 0.9  # Only critical when overloaded
            else:
                return 0.7 - (0.4 * self.params.gamma)
                
        elif self.params.optimization_target == OptimizationTarget.ALERT_VOLUME:
            # Dynamic threshold to maintain constant alert rate
            if hasattr(self, '_recent_alert_rate'):
                if self._recent_alert_rate > 0.1:  # Too many alerts
                    return min(0.95, base_threshold * 1.5)
                else:
                    return max(0.4, base_threshold * 0.8)
            return 0.8
            
        elif self.params.optimization_target == OptimizationTarget.COMPLIANCE:
            # Must investigate anything above minimal threshold
            return 0.15  # Very low threshold for compliance
            
        elif self.params.optimization_target == OptimizationTarget.USER_EXPERIENCE:
            # Only investigate if extremely confident to avoid disruption
            return 0.98 - (0.2 * self.params.gamma)
            
        elif self.params.optimization_target == OptimizationTarget.COST:
            # Cost-benefit analysis
            cost_ratio = self.params.c_auto / self.expected_loss_miss()
            return min(0.95, base_threshold + cost_ratio)
            
        elif self.params.optimization_target == OptimizationTarget.COVERAGE:
            # Low threshold to catch everything possible
            return 0.05 + (0.2 * self.params.gamma)
            
        elif self.params.optimization_target == OptimizationTarget.AUTOMATION:
            # Binary decision - no human escalation
            return 0.5  # Simple threshold
            
        else:  # BALANCED (original)
            # Original gamma-based approach
            if self.params.gamma == 0:
                return 1.0
            else:
                return base_threshold / (1 + 10 * self.params.gamma)


class DynamicEnvironment:
    """Manages dynamic aspects of the simulation environment"""
    
    def __init__(self, params: Parameters):
        self.params = params
        self.current_prior = params.prior
        self.adversary_strength = params.p_evidence_high_given_malicious
        self.queue_size = 0
        
    def update_prior(self, dismiss_rate: float):
        """Update prior based on dismiss rate (dynamic priors)"""
        # If dismissing too much, attackers increase activity
        if dismiss_rate > 0.9:
            self.current_prior = min(self.current_prior * 1.1, 0.1)
        elif dismiss_rate < 0.5:
            self.current_prior = max(self.current_prior * 0.95, 0.01)
            
    def update_adversary(self, dismiss_rate: float):
        """Update adversary behavior based on dismiss rate"""
        if dismiss_rate > self.params.dismiss_threshold:
            # Adversary becomes stealthier
            self.adversary_strength *= self.params.adversary_decay
            self.adversary_strength = max(self.adversary_strength, 0.3)
            
    def get_queue_factor(self) -> float:
        """Calculate queue contention factor"""
        if self.params.queue_capacity == 0:
            return 1.0
        
        utilization = self.queue_size / self.params.queue_capacity
        queue_factor = 1 + self.params.alpha_queue * utilization
        return queue_factor
    
    def generate_incident(self) -> IncidentState:
        """Generate a new incident with current environment parameters"""
        is_malicious = np.random.random() < self.current_prior
        
        if is_malicious:
            # Use current adversary strength
            p_high = self.adversary_strength
            evidence_probs = [p_high, 0.5 * (1 - p_high), 0.5 * (1 - p_high)]
        else:
            # Benign incidents
            evidence_probs = [self.params.p_evidence_high_given_benign, 0.3, 0.6]
            
        evidence_level = np.random.choice(['High', 'Medium', 'Low'], p=evidence_probs)
        
        incident = IncidentState(
            is_malicious=is_malicious,
            evidence_level=evidence_level
        )
        
        return incident