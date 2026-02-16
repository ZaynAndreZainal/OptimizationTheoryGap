# Neural ODE vs. Discrete Baselines on Irregular Biomedical Dynamics

## Overview
This repository contains the reproduction code for **Experiment A.1** from the paper *"Neural Network Dynamics in Biomedical Applications: Reviewing the Gap Between Optimization Instability and Theory-Driven Design"*.

It benchmarks **Continuous-Time Neural Networks (Neural ODEs)** against discrete-time baselines (**LSTMs**, **ODE-RNNs**) on chaotic, irregularly sampled biomedical signals (FitzHugh-Nagumo neuronal dynamics).

The experiment demonstrates the "Discrete-Time Fallacy": standard RNNs fail to generalize when observation intervals are irregular, whereas Neural ODEs learn the underlying continuous vector field.

## Key Features
- **Dynamical Systems:** Simulates FitzHugh-Nagumo (FHN) and Van der Pol oscillators.
- **Irregular Sampling:** Generates training data with randomized time intervals ($dt \sim \mathcal{U}[0.2, 0.8]$).
- **Rigorous Baselines:** Compares Neural ODEs against:
  - **Time-Aware LSTM:** Standard LSTM with time-delta features.
  - **ODE-RNN:** A hybrid architecture (Rubanova et al., NeurIPS 2019) that updates hidden states via ODEs between observations.
- **Statistical Validation:** Runs $n=5$ independent seeds and performs t-tests for significance.

## Requirements
Ensure you have a Python environment (3.8+) with the following dependencies:

```bash
pip install torch numpy matplotlib scipy pandas
```
_Note: A GPU is recommended but not required. The code automatically detects CUDA._

## File Structure
This script is self-contained. It includes:
1. **Data Generation:** RK4 integration of FHN equations.
2. **Model Definitions:** `NeuralODE`, `TimeAwareLSTM`, `ODERNN`.
3. **Training Loop:** Adam optimizer with `ReduceLROnPlateau`.
4. **Evaluation & Plotting:** Generates the 6-panel figure used in the paper (Time domain, Phase space, RMSE comparison).

## Usage
Run the main script to reproduce the experiment: 
```bash
python benchmark_main.py
```

### Expected Output
The script will:
1. Train models across 5 random seeds.
2. Print a statistical summary table (Mean RMSE ± Std).
3. Perform t-tests to verify if Neural ODE performance is significantly better ($p < 0.05$).
4. Save a high-resolution figure: FitzHugh-Nagumo_rigorous.png.

## Configuration
Hyperparameters are defined at the top of the script and match the paper's Appendix:

```python
TRAIN_SAMPLES = 300      # Number of trajectories
SEQ_LEN = 10             # Sequence length per sample
DT_MEAN = 0.5            # Average time step
DT_JITTER = 0.3          # Irregularity magnitude
EXTRAP_DURATION = 2000   # Extrapolation horizon for testing
HIDDEN_DIM = 128         # Model capacity
LEARNING_RATE = 0.0005
EPOCHS = 1500
N_SEEDS = 5              # Number of independent runs
```

## Citation
If you use this code, plase cite the original paper:
@article{zainal2026neural,
  title={Neural Network Dynamics in Biomedical Applications: Reviewing the Gap Between Optimization Instability and Theory-Driven Design},
  author={Zainal, Zayn Andre and Huang, Zhaojing and Aguilar, Isabelle and Kavahei, Omid},
  journal={{TBA)},
  year={2026}
}
