# Edge of Stability Visualization

Multi-seed visualization toolkit for observing Edge of Stability (EoS) phenomena in neural networks.

## Features

- **Multi-seed training** with reproducible checkpoints
- **Loss landscape visualization** via PCA-based 2D/3D projections
- **Sharpness tracking** using power iteration for Hessian eigenvalues
- **Statistical analysis** with 95% confidence intervals across runs

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Single Training Run

```bash
python3 train.py --seed 0
python3 visualize.py --checkpoint 0
```

### Multi-Seed Training (for publication)

Train 5 runs with same base index

```bash
for seed in {0..4}; do
  python3 train.py --seed $seed --base-index 10
done
```

Generate aggreagate visualization

```bash
python3 visualize_multi.py --checkpoint-base 10
```

## Customization

### Adjust 3D viewing angle

```bash
python3 visualize.py --checkpoint 10 --azim-adjust -135 --zoom 1.3
```

### Multi-seed with custom view

```bash
python3 visualize_multi.py --checkpoint-base 10 --azim-adjust -45 --elev-adjust 10
```

## Dataset

Uses MedMNIST PathMNIST (histopathological tissue images, 9 classes).

## Architecture 

3-layer MLP: 2352 -> 256 -> 128 -> 9 (ReLU activations).

## Citation

If you use this code, please cite:
- Cohen et al. (2021) - "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"
- Yang et al. (2023) - "MedMNIST v2"
