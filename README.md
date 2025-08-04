# SOC Simulation: Breaking the "Best-of-Breed Speed" Paradox

A production-ready simulation for Security Operations Center (SOC) operations implementing the Bayesian decision model and reward shaping described in the paper "Breaking the 'Best-of-Breed Speed' Paradox in Security Operations".

## Overview

This simulation demonstrates how a single reward-shaping coefficient (γ) can align AI triage agents with organizational risk, cutting expected losses four-fold while reducing missed attacks from 535 to 25 in 30,000 incidents.

## Features

- **Bayesian Decision Model**: Two-stage triage/investigate decision making with posterior probability calculations
- **Dynamic Environment**: 
  - Adaptive priors based on dismiss rates
  - Adversary behavior adaptation (stealth increases with high dismiss rates)
  - Queue contention modeling with cost increases
- **Heavy-tail Risk Modeling**: Log-normal distribution for breach losses
- **Monte Carlo Simulation**: Full simulation of 30,000 incidents with batch processing
- **Simulated Annealing**: Optimization algorithm to find optimal γ
- **Comprehensive Visualization**: Utility curves, caught/missed crossover plots, and detailed metrics

## Installation

### Option 1: Using pip (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/soc-simulation.git
cd soc-simulation

# Install in development mode
pip install -e .

# Or install with performance extras
pip install -e ".[performance]"
```

### Option 2: Direct requirements installation

```bash
# Clone the repository
git clone https://github.com/yourusername/soc-simulation.git
cd soc-simulation

# Install requirements
pip install -r requirements.txt
```

## Quick Start

### Run all experiments (reproduces paper results)

```bash
python run_experiments.py
```

This will:
1. Run Monte Carlo sweep for γ values from 0.00 to 0.10
2. Execute simulated annealing optimization
3. Generate all visualizations
4. Create a comprehensive report

### Run specific experiments

```bash
# Run only the gamma sweep
python run_experiments.py --sweep-only

# Run only simulated annealing
python run_experiments.py --annealing-only

# Run detailed simulation for specific gamma
python run_experiments.py --gamma 0.05

# Validate results against paper values
python run_experiments.py --validate

# Specify output directory
python run_experiments.py --output-dir custom_results/
```

## Output Files

After running the experiments, you'll find the following in the `results/` directory:

- `parameters.json`: Simulation parameters used
- `sweep_results.csv`: Detailed results for each γ value
- `annealing_results.json`: Optimal γ from simulated annealing
- `utility_gamma.png`: Utility curve plot
- `caught_missed_gamma.png`: Caught vs missed attacks crossover plot
- `results_table.txt`: Summary table similar to paper's Table 1
- `report.md`: Comprehensive summary report
- `batch_metrics_gamma_*.png`: Detailed metrics over time for specific γ
- `loss_distribution_gamma_*.png`: Loss distribution plots

## Key Results

The simulation reproduces the paper's key findings:

| γ    | Utility ↓ | Min/Inc | Inv-rate | Caught | Missed |
|------|-----------|---------|----------|--------|--------|
| 0.00 | -43.3     | 23.9    | 0%       | 0      | 535    |
| 0.05 | -18.4     | 15.2    | 12%      | 245    | 180    |
| **0.09** | **-11.1** | **12.4** | **23%** | **381** | **25** |
| 0.10 | -11.2     | 12.6    | 25%      | 393    | 23     |

## Code Structure

```
soc-simulation/
├── soc_simulation/
│   ├── __init__.py
│   ├── models.py           # Core Bayesian models and environment
│   ├── simulation.py       # Monte Carlo simulation runner
│   └── visualization.py    # Plotting and reporting functions
├── run_experiments.py      # Main experiment runner
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup file
├── README.md              # This file
└── paper.tex              # Original paper (LaTeX)
```

## Model Parameters

The simulation uses the following default parameters from the paper:

- Prior probability of malicious incident: π = 0.02
- Time costs:
  - Skip: 0.5 min
  - Triage: 1.0 min
  - Auto-investigate: 5.0 min
  - Manual investigate: 30.0 min
- Loss distribution: LogNormal(μ=6, σ=1.2)
- Queue contention factor: α = 0.5
- Adversary decay rate: 0.97 when dismiss rate > 80%

## Extending the Simulation

### Adding New Decision Strategies

Modify the `make_triage_decision` method in [`models.py`](soc_simulation/models.py):

```python
def make_triage_decision(self, incident: IncidentState, 
                        queue_factor: float = 1.0) -> DecisionOutcome:
    # Add your custom decision logic here
    pass
```

### Customizing Environment Dynamics

Extend the `DynamicEnvironment` class to add new environmental factors:

```python
class CustomEnvironment(DynamicEnvironment):
    def update_custom_factor(self, metrics):
        # Your custom environment update logic
        pass
```

### Adding New Metrics

Extend the `SimulationMetrics` class in [`simulation.py`](soc_simulation/simulation.py) to track additional metrics.

## Citation

If you use this simulation in your research, please cite:

```bibtex
@article{morpheus2025soc,
  title={Breaking the "Best-of-Breed Speed" Paradox in Security Operations: 
         Bayesian Economics, Reward-Shaping and AI-Agent Evidence},
  author={Analyst, Morpheus T.},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the theoretical framework presented in the paper
- Inspired by real-world SOC operations challenges
- Thanks to the security operations community for insights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
