import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from scipy.interpolate import griddata

parser = argparse.ArgumentParser(description='Visualize multi-seed EoS results')
parser.add_argument('--checkpoint-base', type=int, required=True,
                    help='Base checkpoint index (e.g., 46 for checkpoint_46_seed*.pkl)')
parser.add_argument('--azim-adjust', type=float, default=0)
parser.add_argument('--elev-adjust', type=float, default=0)
parser.add_argument('--zoom', type=float, default=1.2)
args = parser.parse_args()

CHECKPOINTS_DIR = Path('./checkpoints')
OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all seed checkpoints
checkpoint_pattern = f'checkpoint_{args.checkpoint_base}_seed*.pkl'
checkpoint_files = sorted(CHECKPOINTS_DIR.glob(checkpoint_pattern))

if len(checkpoint_files) == 0:
  raise FileNotFoundError(f"No checkpoints found matching {checkpoint_pattern}")

print(f"Found {len(checkpoint_files)} checkpoint files")

# Load all runs
all_metrics = []
all_checkpoints = []
all_hyperparams = []

for ckpt_file in checkpoint_files:
  with open(ckpt_file, 'rb') as f:
    data = pickle.load(f)
    all_metrics.append(data['metrics'])
    all_checkpoints.append(data['checkpoints'])
    all_hyperparams.append(data['hyperparameters'])
  seed = ckpt_file.stem.split('_seed')[1]
  print(f"  Loaded seed {seed}")

# Extract common hyperparameters
LEARNING_RATE = all_hyperparams[0]['learning_rate']
EPOCHS = all_hyperparams[0]['epochs']
STABILITY_LIMIT = all_hyperparams[0]['stability_limit']
BATCH_SIZE = all_hyperparams[0]['batch_size']
n_channels = all_hyperparams[0].get('n_channels', 1)
n_classes = all_hyperparams[0].get('n_classes', 10)

# Aggregate metrics across seeds
epochs = all_metrics[0]['epoch'][1:]  # Skip epoch 0
n_epochs = len(epochs)
n_seeds = len(all_metrics)

loss_matrix = np.zeros((n_seeds, n_epochs))
sharpness_matrix = np.zeros((n_seeds, n_epochs))

for i, metrics in enumerate(all_metrics):
  loss_matrix[i] = metrics['loss'][1:]
  sharpness_matrix[i] = metrics['sharpness'][1:]

# Compute statistics
loss_mean = loss_matrix.mean(axis=0)
loss_std = loss_matrix.std(axis=0)
loss_ci_lower = loss_mean - 1.96 * loss_std / np.sqrt(n_seeds)  # 95% CI
loss_ci_upper = loss_mean + 1.96 * loss_std / np.sqrt(n_seeds)

sharpness_mean = sharpness_matrix.mean(axis=0)
sharpness_std = sharpness_matrix.std(axis=0)
sharpness_ci_lower = sharpness_mean - 1.96 * sharpness_std / np.sqrt(n_seeds)
sharpness_ci_upper = sharpness_mean + 1.96 * sharpness_std / np.sqrt(n_seeds)

# Find median run (by final loss)
final_losses = loss_matrix[:, -1]
median_idx = np.argsort(final_losses)[len(final_losses) // 2]
print(f"Using seed {median_idx} for landscape visualization (median final loss)")

# Create figure
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], width_ratios=[1.9, 1.5, 1.6],
                      hspace=0.4, wspace=0.35,
                      left=0.05, right=0.95, top=0.95, bottom=0.05)

# Panel A: Multi-seed loss and sharpness
print("Creating Panel A...")

# Color scheme
SHARPNESS_COLOR = '#17BECF'  # Cyan
STABILITY_LINE_COLOR = '#1A237E'  # Deep indigo
INSTABILITY_SHADE_COLOR = '#5C6BC0'  # Light indigo
INSTABILITY_ALPHA = 0.18

ax_loss = fig.add_subplot(gs[0, :])

# Create plasma colormap for trajectory
colors_plasma = plt.cm.plasma(np.linspace(0, 1, len(epochs)))

# Plot individual loss runs (thin, transparent, plasma gradient)
for i in range(n_seeds):
    for j in range(len(epochs) - 1):
        ax_loss.plot(epochs[j:j+2], loss_matrix[i, j:j+2],
                     color=colors_plasma[j], alpha=0.15, linewidth=1)

# Plot mean loss (thick, plasma gradient)
for j in range(len(epochs) - 1):
    ax_loss.plot(epochs[j:j+2], loss_mean[j:j+2],
                 color=colors_plasma[j], linewidth=2.5, zorder=10)

# Confidence interval
ax_loss.fill_between(epochs, loss_ci_lower, loss_ci_upper,
                      color='gray', alpha=0.15, zorder=5)

ax_loss.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
ax_loss.set_xlabel('Epochs', fontsize=12)
ax_loss.tick_params(axis='y', labelsize=11)
ax_loss.grid(alpha=0.3)
ax_loss.set_xlim(left=1)
ax_loss.set_ylim(bottom=0)

# Sharpness on secondary axis
ax_sharp = ax_loss.twinx()

# Individual sharpness runs
for i in range(n_seeds):
    ax_sharp.plot(epochs, sharpness_matrix[i], color=SHARPNESS_COLOR, alpha=0.2, linewidth=1)

# Mean sharpness
ax_sharp.plot(epochs, sharpness_mean, color=SHARPNESS_COLOR, linewidth=2.5,
              label=f'Sharpness (n={n_seeds})', zorder=10)

# Confidence interval
ax_sharp.fill_between(epochs, sharpness_ci_lower, sharpness_ci_upper,
                       color=SHARPNESS_COLOR, alpha=0.15, zorder=5)

ax_sharp.set_ylabel('Sharpness (Top Eigenvalue)', color=SHARPNESS_COLOR,
                    fontsize=12, fontweight='bold')
ax_sharp.tick_params(axis='y', labelcolor=SHARPNESS_COLOR, labelsize=11)
ax_sharp.set_ylim(bottom=0)

# Stability limit and instability zone
ax_sharp.axhline(y=STABILITY_LIMIT, color=STABILITY_LINE_COLOR, linestyle='--', linewidth=2.5,
                label=f'Stability Limit (2/η = {STABILITY_LIMIT:.1f})', zorder=8)

max_sharpness_display = sharpness_matrix.max() * 1.05
ax_sharp.axhspan(STABILITY_LIMIT, max_sharpness_display,
                 alpha=INSTABILITY_ALPHA, color=INSTABILITY_SHADE_COLOR,
                 label='Instability Zone', zorder=0)
ax_sharp.set_ylim(bottom=0, top=max_sharpness_display)
ax_sharp.set_xlim(left=1)

# Custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color=colors_plasma[0], linewidth=2.5, label=f'Training Loss (n={n_seeds})'),
    Patch(facecolor='gray', alpha=0.15, label='95% CI (Loss)'),
    Line2D([0], [0], color=SHARPNESS_COLOR, linewidth=2.5, label=f'Sharpness (n={n_seeds})'),
    Patch(facecolor=SHARPNESS_COLOR, alpha=0.15, label='95% CI (Sharpness)'),
    Line2D([0], [0], color=STABILITY_LINE_COLOR, linestyle='--', linewidth=2.5,
           label=f'Stability Limit (2/η = {STABILITY_LIMIT:.1f})'),
    Patch(facecolor=INSTABILITY_SHADE_COLOR, alpha=INSTABILITY_ALPHA, label='Instability Zone')
]

ax_sharp.legend(handles=legend_elements,
                loc='upper left', bbox_to_anchor=(1.02, 1),
                fontsize=10, frameon=True, fancybox=True)
ax_loss.text(-0.04, 1.05, 'a', transform=ax_loss.transAxes,
             fontsize=16, fontweight='bold', va='top')

# ==========================================
# PANELS B & C: LANDSCAPE FROM MEDIAN RUN
# ==========================================

# Load median checkpoint
median_checkpoint_file = checkpoint_files[median_idx]
with open(median_checkpoint_file, 'rb') as f:
  median_data = pickle.load(f)

median_checkpoints = median_data['checkpoints']
median_eigen_data = median_data['eigen_data'].to(torch.device('cpu'))
median_eigen_target = median_data['eigen_target'].to(torch.device('cpu'))


# Recreate model for landscape computation
class SimpleMLP(nn.Module):
  def __init__(self, n_channels=3, n_classes=9):
    super(SimpleMLP, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(n_channels * 28 * 28, 256)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(256, 128)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(128, n_classes)

  def forward(self, x):
    x = self.flatten(x)
    x = self.relu1(self.fc1(x))
    x = self.relu2(self.fc2(x))
    return self.fc3(x)


landscape_model = SimpleMLP(n_channels=n_channels, n_classes=n_classes)
landscape_model.load_state_dict(median_data['model_state_dict'])
landscape_criterion = nn.CrossEntropyLoss()


# Helper functions
def compute_loss_landscape_2d(model, data, target, criterion, center_params, dir1, dir2,
                              alpha_min, alpha_max, beta_min, beta_max, grid_points=40):
  alpha_vals = np.linspace(alpha_min, alpha_max, grid_points)
  beta_vals = np.linspace(beta_min, beta_max, grid_points)
  X, Y = np.meshgrid(alpha_vals, beta_vals)
  Z = np.zeros_like(X)
  original_params = [p.clone() for p in model.parameters()]

  for i, alpha in enumerate(alpha_vals):
    for j, beta in enumerate(beta_vals):
      with torch.no_grad():
        for p, center, d1, d2 in zip(model.parameters(), center_params, dir1, dir2):
          p.copy_(center + alpha * d1 + beta * d2)
      model.eval()
      with torch.no_grad():
        Z[j, i] = criterion(model(data), target).item()

  with torch.no_grad():
    for p, orig in zip(model.parameters(), original_params):
      p.copy_(orig)
  return X, Y, Z


def project_params_onto_plane(params, center, dir1, dir2):
  diff = [p - c for p, c in zip(params, center)]
  alpha = sum(torch.sum(d * d1) for d, d1 in zip(diff, dir1)).item()
  beta = sum(torch.sum(d * d2) for d, d2 in zip(diff, dir2)).item()
  return alpha, beta


def unflatten_vector(flat_vec, shapes, numels):
  result = []
  offset = 0
  for shape, numel in zip(shapes, numels):
    result.append(flat_vec[offset:offset + numel].reshape(shape))
    offset += numel
  return result


# Compute gradient-aligned directions
print("Computing landscape directions...")
initial_params_flat = torch.cat([p.flatten() for p in median_checkpoints[0]['params']])
final_params_flat = torch.cat([p.flatten() for p in median_checkpoints[-1]['params']])
dir1_flat = (final_params_flat - initial_params_flat)
dir1_flat = dir1_flat / torch.norm(dir1_flat)

# PCA for orthogonal direction
checkpoint_vectors = [torch.cat([p.flatten() for p in ckpt['params']]) for ckpt in median_checkpoints]
checkpoint_matrix = torch.stack(checkpoint_vectors)
mean_params = checkpoint_matrix.mean(dim=0)
centered_matrix = checkpoint_matrix - mean_params
U, S, Vt = torch.linalg.svd(centered_matrix.T, full_matrices=False)
pc2_candidate = U[:, 1]
dir2_flat = pc2_candidate - (torch.dot(pc2_candidate, dir1_flat) * dir1_flat)
dir2_flat = dir2_flat / torch.norm(dir2_flat)

variance_explained = (S ** 2) / (S ** 2).sum()
pca_var1 = variance_explained[0].item() * 100
pca_var2 = variance_explained[1].item() * 100

# Unflatten
params_list = [p for p in landscape_model.parameters() if p.requires_grad]
param_shapes = [p.shape for p in params_list]
param_numel = [p.numel() for p in params_list]
dir1 = unflatten_vector(dir1_flat, param_shapes, param_numel)
dir2 = unflatten_vector(dir2_flat, param_shapes, param_numel)
center_params = median_checkpoints[0]['params']

# Compute trajectory bounds
trajectory_alphas_temp = []
trajectory_betas_temp = []
for ckpt in median_checkpoints:
  alpha, beta = project_params_onto_plane(ckpt['params'], center_params, dir1, dir2)
  trajectory_alphas_temp.append(alpha)
  trajectory_betas_temp.append(beta)

alpha_min, alpha_max = min(trajectory_alphas_temp), max(trajectory_alphas_temp)
beta_min, beta_max = min(trajectory_betas_temp), max(trajectory_betas_temp)
padding_factor = 0.4
alpha_min_padded = alpha_min - (alpha_max - alpha_min) * padding_factor
alpha_max_padded = alpha_max + (alpha_max - alpha_min) * padding_factor
beta_min_padded = beta_min - (beta_max - beta_min) * padding_factor
beta_max_padded = beta_max + (beta_max - beta_min) * padding_factor

# Compute loss landscape
print("Computing loss landscape...")
X, Y, Z = compute_loss_landscape_2d(
  landscape_model, median_eigen_data, median_eigen_target, landscape_criterion,
  center_params, dir1, dir2,
  alpha_min_padded, alpha_max_padded, beta_min_padded, beta_max_padded, grid_points=40)

# Project trajectory
trajectory_coords = []
for ckpt in median_checkpoints:
  alpha, beta = project_params_onto_plane(ckpt['params'], center_params, dir1, dir2)
  with torch.no_grad():
    for p, ckpt_p in zip(landscape_model.parameters(), ckpt['params']):
      p.copy_(ckpt_p)
  landscape_model.eval()
  with torch.no_grad():
    actual_loss = landscape_criterion(landscape_model(median_eigen_data), median_eigen_target).item()
  trajectory_coords.append({
    'epoch': ckpt['epoch'], 'alpha': alpha, 'beta': beta,
    'loss': actual_loss, 'sharpness': ckpt['sharpness']
  })

traj_alphas = [t['alpha'] for t in trajectory_coords]
traj_betas = [t['beta'] for t in trajectory_coords]
traj_losses = [t['loss'] for t in trajectory_coords]
traj_epochs = [t['epoch'] for t in trajectory_coords]

# Panel B: 3D Landscape
ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
surf = ax_3d.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.75,
                          edgecolor='none', antialiased=True, zorder=1)

colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory_coords)))

# Trajectory with surface offset
surface_losses = [griddata((X.flatten(), Y.flatten()), Z.flatten(), (a, b), method='linear')
                  for a, b in zip(traj_alphas, traj_betas)]
traj_z_coords = [sl * 1.15 for sl in surface_losses]

for i in range(len(trajectory_coords) - 1):
  ax_3d.plot(traj_alphas[i:i + 2], traj_betas[i:i + 2],
             [traj_z_coords[i], traj_z_coords[i + 1]],
             color=colors[i], linewidth=2.5, zorder=10)

# Key markers
key_epochs_idx = [0] + list(range(10, len(trajectory_coords), 10)) + [len(trajectory_coords) - 1]
key_alphas = [traj_alphas[i] for i in key_epochs_idx]
key_betas = [traj_betas[i] for i in key_epochs_idx]
key_z = [traj_z_coords[i] for i in key_epochs_idx]
ax_3d.scatter(key_alphas, key_betas, key_z, c=[traj_epochs[i] for i in key_epochs_idx],
              cmap='plasma', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=15)
ax_3d.scatter([traj_alphas[0]], [traj_betas[0]], [traj_z_coords[0]],
              color='lime', s=400, marker='o', edgecolors='black', linewidths=3,
              label='Start (E0)', zorder=20)
ax_3d.scatter([traj_alphas[-1]], [traj_betas[-1]], [traj_z_coords[-1]],
              color='red', s=400, marker='*', edgecolors='black', linewidths=3,
              label=f'End (E{EPOCHS})', zorder=20)

# View angle
azim_base = np.degrees(np.arctan2(traj_betas[-1] - traj_betas[0], traj_alphas[-1] - traj_alphas[0]))
elev = (20 if (traj_losses[0] - traj_losses[-1]) > 1.0 else 30) + args.elev_adjust
azim = (azim_base % 360) + args.azim_adjust

ax_3d.set_axis_off()
ax_3d.set_box_aspect(aspect=None, zoom=args.zoom)
ax_3d.margins(0, 0, 0)
# After ax_3d plotting, update position
pos = ax_3d.get_position()
ax_3d.set_position([pos.x0, pos.y0 + 0.02, pos.width * 1.05, pos.height * 1.05])  # Shift up slightly
ax_3d.view_init(elev=elev, azim=azim)
ax_3d.text2D(-0.1, 1.05, 'b', transform=ax_3d.transAxes,
             fontsize=16, fontweight='bold', va='top')

# Panel C: 2D Contour
ax_contour = fig.add_subplot(gs[1, 1])
contour = ax_contour.contourf(X, Y, Z, levels=25, cmap='coolwarm', alpha=0.9)
ax_contour.contour(X, Y, Z, levels=12, colors='black', alpha=0.3, linewidths=0.8)

# FORCE SQUARE ASPECT RATIO
# ax_contour.set_aspect('equal', adjustable='box')  # ADD THIS LINE RIGHT HERE

# Trajectory
for i in range(len(trajectory_coords) - 1):
    ax_contour.plot(traj_alphas[i:i+2], traj_betas[i:i+2],
                    color=colors[i], linewidth=2.5, zorder=10)

ax_contour.scatter(key_alphas, key_betas, c=[traj_epochs[i] for i in key_epochs_idx],
                   cmap='plasma', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=15)
ax_contour.scatter(traj_alphas[0], traj_betas[0], color='lime', s=400,
                   marker='o', edgecolors='black', linewidths=3, label='Start (E0)', zorder=20)
ax_contour.scatter(traj_alphas[-1], traj_betas[-1], color='red', s=400,
                   marker='*', edgecolors='black', linewidths=3, label=f'End (E{EPOCHS})', zorder=20)

cbar = plt.colorbar(contour, ax=ax_contour, fraction=0.046, pad=0.04)
cbar.set_label('Loss', fontsize=11, fontweight='bold')

ax_contour.set_xlabel('PC1 (Primary Direction)', fontsize=10)
ax_contour.set_ylabel('PC2 (Secondary Direction)', fontsize=10)
ax_contour.legend(loc='upper left', fontsize=9)
ax_contour.grid(alpha=0.3)
ax_contour.text(-0.15, 1.05, 'c', transform=ax_contour.transAxes,
                fontsize=16, fontweight='bold', va='top')

# Panel D: Aggregate statistics (Two-column layout)
ax_summary = fig.add_subplot(gs[1, 2])
ax_summary.axis('off')

# Compute aggregate stats
max_sharpnesses = [max(m['sharpness']) for m in all_metrics]
final_sharpnesses = [m['sharpness'][-1] for m in all_metrics]
initial_losses = [m['loss'][1] for m in all_metrics]
final_losses_all = [m['loss'][-1] for m in all_metrics]

# LEFT COLUMN
left_text = f"""TRAINING SETUP
━━━━━━━━━━━━━━━
Architecture
  3-layer MLP
  {n_channels * 28 * 28}→256→128→{n_classes}

Dataset
  PathMNIST (n={n_seeds})

Optimizer
  SGD (no momentum)
  LR: {LEARNING_RATE}
  Batch: {BATCH_SIZE}
  Epochs: {EPOCHS}

LANDSCAPE
━━━━━━━━━━━━━━━
PCA Variance
  PC1: {pca_var1:.1f}%
  PC2: {pca_var2:.1f}%

Median Run
  Seed {median_idx}"""

# RIGHT COLUMN
right_text = f"""METRICS (n={n_seeds})
━━━━━━━━━━━━━━━
Stability Limit
  2/η = {STABILITY_LIMIT:.2f}

Peak Sharpness
  {np.mean(max_sharpnesses):.1f} ± {np.std(max_sharpnesses):.1f}
  Range: [{np.min(max_sharpnesses):.1f}, {np.max(max_sharpnesses):.1f}]

Final Sharpness
  {np.mean(final_sharpnesses):.1f} ± {np.std(final_sharpnesses):.1f}

Initial Loss
  {np.mean(initial_losses):.3f} ± {np.std(initial_losses):.3f}

Final Loss
  {np.mean(final_losses_all):.3f} ± {np.std(final_losses_all):.3f}

Reduction
  {((np.mean(initial_losses) - np.mean(final_losses_all)) / np.mean(initial_losses) * 100):.1f}%"""

# Draw left column at x=0.02 (moved left)
ax_summary.text(0.02, 0.95, left_text,
                transform=ax_summary.transAxes,
                fontsize=14,  # Slightly smaller
                verticalalignment='top',
                fontfamily='monospace',
                linespacing=1.3)

# Draw right column at x=0.58 (moved right)
ax_summary.text(0.58, 0.95, right_text,
                transform=ax_summary.transAxes,
                fontsize=14,  # Slightly smaller
                verticalalignment='top',
                fontfamily='monospace',
                linespacing=1.3)

ax_summary.text(-0.12, 1.05, 'd', transform=ax_summary.transAxes,
                fontsize=16, fontweight='bold', va='top')

# Save
filename = OUTPUT_DIR / f'eos_multiseed_figure_{args.checkpoint_base}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✓ Multi-seed figure saved: {filename}")