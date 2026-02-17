# OptimizationTheoryGap: Neural Network Dynamics in Biomedical Applications

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation and experimental data for the paper:

**"Neural Network Dynamics in Biomedical Applications: Reviewing the Gap Between Optimization Instability and Theory-Driven Design"**
*Zayn Andre Zainal, Zhaojing Huang, Isabelle Aguilar, Omid Kavahei*

## Abstract

> Artificial Intelligence has undeniably advanced biomedical engineering, particularly in diagnostics and pharmacokinetics. Yet, the field remains limited by a reliance on static mapping approaches which struggle to capture the complex, shifting nature of biological systems. To bridge the gap between theory and scalable application, this review proposes viewing biomedical AI through the lens of dynamical systems theory. By applying principles such as Lyapunov stability and attractor geometry, we can model physiological states, like homeostasis or epilepsy, as distinct topological behaviours rather than simple classification labels. Modern architectures are already evolving to reflect this, moving from discrete-time recurrent neural networks (RNNs) to continuous-time Neural Ordinary Differential Equations (ODEs) and Spiking Neural Networks (SNNs). However, theoretical challenges remain. We specifically address the instability of gradient-based optimization at the “Edge of Stability” (EoS), a phenomenon that risks model reliability and causes hallucinations in safety-critical imaging. Ultimately, sustainable progress depends not just on more data, but on establishing a unified theory that aligns algorithmic design with the physical realities of biological mechanisms.

---

## Repository Structure

The codebase is organized by experiment, corresponding directly to the Appendices in the paper:

```text
OptimizationTheoryGap/
├── experiments/
│   ├── 01_irregular_sampling/   # [Appendix A.1] Neural ODEs vs. LSTMs on FHN Dynamics
│   └── 02_edge_of_stability/    # [Appendix A.2] Loss Landscape & Sharpness Visualization
├── requirements.txt             # Shared dependencies for all experiments
└── README.md                    # You are here
```

### 1. Irregular Sampling Benchmark (Appendix A.1)
_Located in: `experiments/01_irregular_sampling/`_

Demonstrates the "Discrete-Time Fallacy" by benchmarking Neural ODEs against Time-Aware LSTMs and ODE-RNNs on chaotic FitzHugh-Nagumo neuronal dynamics.
- **Key Finding**: Discrete RNNs accumulate significant drift when data is sparse or irregularly sampled, whereas Neural ODEs learn the continuous vector field.
- **Metrics**: RMSE, Phase Space Trajectory Divergence.

### 2. Edge of Stability Visualization (Appendix A.2)
_Located in: `experiments/02_edge_of_stability/`_
Investigates optimization instability in deep networks trained on biomedical data (MedMNIST).
- **Key Finding**: Modern optimizers do not converge to flat minima but "surf" the walls of high-curvature valleys ($\lambda_{max} > 2/\eta$), creating a risk of fragility in safety-critical deployments.
- **Visualization**: 3D PCA projections of the loss landscape and Hessian spectral estimation.

## **Getting Started**
### Prerequisites
Ensure you have Python 3.8+ installed. We recommend creating a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (PyTorch, SciPy, Matplotlib, MedMNIST)
pip install -r requirements.txt
```
_Note: For GPU acceleration, please install the appropriate CUDA version of PyTorch from pytorch.org before running the requirements file._

### Experiment A.1: Irregular Sampling (FitzHugh-Nagumo)
To reproduce the time-series benchmarks and generate Figure 2 from the paper:
```bash
cd experiments/01_irregular_sampling
python3 benchmark_main.py
```
_Output: `FitzHugh-Nagumo_rigorous.png` abd statistical summary table._

### Experiment A.2: Edge of Stability (MedMNIST)
To reproduce the loss landscape visualization and Figure 3 from the paper:
```bash
cd experiments/02_edge_of_stability

# 1. Train the model (run for multiple seeds for confidence intervals)
python3 train_eos.py --seed 0

# 2. Visualize the trajectory
python3 visualize.py --checkpoint 0

# 3. Visualize multi-seed trajectory (publication figure)
python3 visualize_multi.py --seed 0
```

_Output: `figures/eos_single_0.png` showing the 3D optimization path._

## Citation
If you use this code or our findings in your research, please cite the paper:

```bibtex
@article{zainal2026neural,
  title={Neural Network Dynamics in Biomedical Applications: Reviewing the Gap Between Optimization Instability and Theory-Driven Design},
  author={Zainal, Zayn Andre and Huang, Zhaojing and Aguilar, Isabelle and Kavahei, Omid},
  journal={(TBA)},
  year={2026},
  institution={School of Biomedical Engineering, University of Sydney}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Corresponding author e-mail**: Andre Zainal (andre.zainal@sydney.edu.au)
