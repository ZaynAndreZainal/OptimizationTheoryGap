# Edge of Stability Visualization
A multi-seed visualization toolkit for observing "Edge of Stability" phenomena in neural networks trained on biomedical data (MedMNIST).

## Overview

This experiment reproduces **Appendix A.2** of the paper. It demonstrates that neural networks optimizing on complex biomedical loss landscapes do not converge to a flat local minimum. Instead, they enter a chaotic "Edge of Stability" regime where the sharpness of the loss function ($\lambda_{max}$) hovers just above the stability threshold ($2/\eta$).

## Features

- **Multi-seed Training:** Statistical validation across 5 independent seeds.
- **Hessian Analysis:** Tracks the maximum eigenvalue ($\lambda_{max}$) using Power Iteration to quantify sharpness.
- **Loss Landscape Visualization:** Generates 2D and 3D PCA projections of the optimization trajectory.
- **MedMNIST Integration:** Uses *PathMNIST* (histopathological tissue patches) to simulate realistic biomedical classification tasks.

## Installation

This experiment relies on the shared environment defined in the repository root.

```bash
# Navigate to root and install dependencies
cd ../../
pip install -r requirements.txt

# Navigate back to this experiment
cd experiments/02_edge_of_stability
```

## Usage

### 1. Single Training Run
Train a single model and visualize its specific trajectory.

```bash
# Train (Save to checkpoints/run_0.pt)
python train_eos.py --seed 0

# Visualize (Generates figures/eos_single_0.png)
python visualize_loss.py --checkpoint 0
```

### 2. Reproduce Paper Figure (Multi-Seed)

To generate the aggregated figure with confidence intervals (as seen in the paper):

#### Step A: Train 5 independent models

```bash
# Bash loop (Linux/Mac)
for seed in {0..4}; do
  python train_eos.py --seed $seed --base-index 10
done

# PowerShell loop (Windows)
# for ($i=0; $i -lt 5; $i++) { python train_eos.py --seed $i --base-index 10 }
```

#### Step B: Generate Aggregate Visualization

```bash
python3 visualize_multi.py --checkpoint-base 10
```

## Customization
You can adjust the camera angles for the 3D plots to better reveal the "surfing" behavior along the valley walls.

### Adjust 3D viewing angle

```bash
python visualize.py --checkpoint 10 --azim-adjust -135 --zoom 1.3
```

### Multi-seed with custom view

```bash
python visualize_multi.py --checkpoint-base 10 --azim-adjust -45 --elev-adjust 10
```

## Experiment Details

### Architecture 
- **Model**: 3-layer MLP (ReLU activations)
- **Dimensions**: $2352 \to 256 \to 128 \to 9$
- **Input**: Flattened $28 \times 28$ MedMNIST images (PathMNIST)

### Dataset
**PathMNIST**: A dataset of colorectal cancer histology slides, containing 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images.

## Citation
If you use this code, please cite the main paper and the underlying datasets/theories:
- **Primary Paper**: Zainal et al. (2025) - "Neural Network Dynamics in Biomedical Applications"
- **Edge of Stability Theory**: Cohen et al. (2021) - "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"
- **Dataset**: Yang et al. (2023) - "MedMNIST v2 - A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Analysis"

## Example Output
![Multi-seed EoS Visualization](figures/eos_multiseed_figure_10.png)
