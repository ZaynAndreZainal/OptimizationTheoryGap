import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import argparse
from scipy.interpolate import griddata

# ==========================================
# 0. COMMAND LINE ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description='Visualize EoS training results')
parser.add_argument('--azim-adjust', type=float, default=0,
                    help='Azimuth angle adjustment in degrees (default: 0)')
parser.add_argument('--elev-adjust', type=float, default=0,
                    help='Elevation angle adjustment in degrees (default: 0)')
parser.add_argument('--zoom', type=float, default=1.2,
                    help='3D plot zoom factor (default: 1.2)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Checkpoint file to load (default: latest checkpoint_*.pkl)')
args = parser.parse_args()

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHECKPOINTS_DIR = Path('./checkpoints')

if args.checkpoint:
  # Handle both full filename and just the index number
  checkpoint_arg = args.checkpoint

  # If user passed just a number, construct the filename
  if checkpoint_arg.isdigit():
    checkpoint_arg = f'checkpoint_{checkpoint_arg}.pkl'

  # If user passed filename without .pkl extension, add it
  if not checkpoint_arg.endswith('.pkl'):
    checkpoint_arg = f'{checkpoint_arg}.pkl'

  CHECKPOINT_FILE = str(CHECKPOINTS_DIR / checkpoint_arg)

  try:
    CHECKPOINT_INDEX = int(Path(checkpoint_arg).stem.replace('checkpoint_', '').split('_seed')[0])
  except:
    CHECKPOINT_INDEX = 0
else:
  # Auto-find checkpoint with largest index in checkpoints/
  existing = list(CHECKPOINTS_DIR.glob('checkpoint_*.pkl'))
  if not existing:
    raise FileNotFoundError(f"No checkpoint files found in {CHECKPOINTS_DIR}!")

  checkpoint_indices = []
  for f in existing:
    try:
      idx = int(f.stem.replace('checkpoint_', '').split('_seed')[0])
      checkpoint_indices.append((idx, str(f)))
    except:
      continue

  if not checkpoint_indices:
    raise FileNotFoundError("No valid checkpoint files found!")

  CHECKPOINT_INDEX, CHECKPOINT_FILE = max(checkpoint_indices, key=lambda x: x[0])

OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")
print(f"Checkpoint index: {CHECKPOINT_INDEX}")
print(f"Loading from: {CHECKPOINT_FILE}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================
with open(CHECKPOINT_FILE, 'rb') as f:
  training_data = pickle.load(f)

checkpoints = training_data['checkpoints']
metrics = training_data['metrics']
hyperparams = training_data['hyperparameters']
eigen_data = training_data['eigen_data'].to(DEVICE)
eigen_target = training_data['eigen_target'].to(DEVICE)

LEARNING_RATE = hyperparams['learning_rate']
EPOCHS = hyperparams['epochs']
STABILITY_LIMIT = hyperparams['stability_limit']
BATCH_SIZE = hyperparams['batch_size']
n_channels = hyperparams.get('n_channels', 1)
n_classes = hyperparams.get('n_classes', 10)

print(f"Loaded {len(checkpoints)} checkpoints from {EPOCHS} epochs")


# ==========================================
# 3. RECREATE MODEL
# ==========================================
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


model = SimpleMLP(n_channels=n_channels, n_classes=n_classes).to(DEVICE)
model.load_state_dict(training_data['model_state_dict'])
criterion = nn.CrossEntropyLoss()


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def compute_loss_landscape_2d_asymmetric(model, data, target, criterion,
                                         center_params, dir1, dir2,
                                         alpha_min, alpha_max, beta_min, beta_max,
                                         grid_points=40):
  """Compute loss on 2D grid with asymmetric bounds."""
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
        output = model(data)
        Z[j, i] = criterion(output, target).item()

  with torch.no_grad():
    for p, orig in zip(model.parameters(), original_params):
      p.copy_(orig)
  return X, Y, Z


def project_params_onto_plane(params, center, dir1, dir2):
  """Project parameters onto 2D plane."""
  diff = [p - c for p, c in zip(params, center)]
  alpha = sum(torch.sum(d * d1) for d, d1 in zip(diff, dir1)).item()
  beta = sum(torch.sum(d * d2) for d, d2 in zip(diff, dir2)).item()
  return alpha, beta


def unflatten_vector(flat_vec, shapes, numels):
  """Unflatten vector to parameter list."""
  result = []
  offset = 0
  for shape, numel in zip(shapes, numels):
    result.append(flat_vec[offset:offset + numel].reshape(shape))
    offset += numel
  return result


# ==========================================
# 5. COMPUTE LOSS LANDSCAPE
# ==========================================
print("\n" + "=" * 60)
print("GENERATING COMBINED VISUALIZATION")
print("=" * 60)
print("Step 1: Computing gradient-aligned directions...")

# Get initial and final parameters
initial_params_flat = torch.cat([p.flatten() for p in checkpoints[0]['params']])
final_params_flat = torch.cat([p.flatten() for p in checkpoints[-1]['params']])

# Direction 1: Initial → Final (captures loss descent)
dir1_flat = final_params_flat - initial_params_flat
dir1_flat = dir1_flat / torch.norm(dir1_flat)

# Direction 2: Compute PCA just to get an orthogonal direction
checkpoint_vectors = []
for ckpt in checkpoints:
  flat_params = torch.cat([p.flatten() for p in ckpt['params']])
  checkpoint_vectors.append(flat_params)

checkpoint_matrix = torch.stack(checkpoint_vectors)
mean_params = checkpoint_matrix.mean(dim=0)
centered_matrix = checkpoint_matrix - mean_params
U, S, Vt = torch.linalg.svd(centered_matrix.T, full_matrices=False)
pc2_candidate = U[:, 1]

# Orthogonalize using Gram-Schmidt
dir2_flat = pc2_candidate - (torch.dot(pc2_candidate, dir1_flat) * dir1_flat)
dir2_flat = dir2_flat / torch.norm(dir2_flat)

variance_explained = (S ** 2) / (S ** 2).sum()
print(f"  Dir1: Initial → Final (loss descent direction)")
print(f"  Dir2: Orthogonal (PCA PC2, orthogonalized)")
print(f"  PCA variance: PC1 {variance_explained[0].item() * 100:.1f}%, PC2 {variance_explained[1].item() * 100:.1f}%")

# Unflatten vectors
params_list = [p for p in model.parameters() if p.requires_grad]
param_shapes = [p.shape for p in params_list]
param_numel = [p.numel() for p in params_list]

dir1 = unflatten_vector(dir1_flat, param_shapes, param_numel)
dir2 = unflatten_vector(dir2_flat, param_shapes, param_numel)

# Use initial parameters as center
center_params = [p.to(DEVICE) for p in checkpoints[0]['params']]

print("Step 2: Computing trajectory bounds...")
trajectory_alphas_temp = []
trajectory_betas_temp = []
for ckpt in checkpoints:
  params_on_device = [p.to(DEVICE) for p in ckpt['params']]
  alpha, beta = project_params_onto_plane(
    params_on_device, center_params,
    [d.to(DEVICE) for d in dir1],
    [d.to(DEVICE) for d in dir2]
  )
  trajectory_alphas_temp.append(alpha)
  trajectory_betas_temp.append(beta)

alpha_min, alpha_max = min(trajectory_alphas_temp), max(trajectory_alphas_temp)
beta_min, beta_max = min(trajectory_betas_temp), max(trajectory_betas_temp)
alpha_range = alpha_max - alpha_min
beta_range = beta_max - beta_min

padding_factor = 0.4
alpha_min_padded = alpha_min - alpha_range * padding_factor
alpha_max_padded = alpha_max + alpha_range * padding_factor
beta_min_padded = beta_min - beta_range * padding_factor
beta_max_padded = beta_max + beta_range * padding_factor

print("Step 3: Computing loss landscape...")
X, Y, Z = compute_loss_landscape_2d_asymmetric(
  model, eigen_data, eigen_target, criterion,
  center_params,
  [d.to(DEVICE) for d in dir1],
  [d.to(DEVICE) for d in dir2],
  alpha_min=alpha_min_padded,
  alpha_max=alpha_max_padded,
  beta_min=beta_min_padded,
  beta_max=beta_max_padded,
  grid_points=40)

print("Step 4: Projecting trajectory...")
trajectory_coords = []
for ckpt in checkpoints:
  params_on_device = [p.to(DEVICE) for p in ckpt['params']]
  alpha, beta = project_params_onto_plane(
    params_on_device, center_params,
    [d.to(DEVICE) for d in dir1],
    [d.to(DEVICE) for d in dir2]
  )

  # Compute ACTUAL loss at checkpoint (not interpolated)
  with torch.no_grad():
    for p, ckpt_p in zip(model.parameters(), params_on_device):
      p.copy_(ckpt_p)
  model.eval()
  with torch.no_grad():
    output = model(eigen_data)
    actual_loss = criterion(output, eigen_target).item()

  trajectory_coords.append({
    'epoch': ckpt['epoch'],
    'alpha': alpha,
    'beta': beta,
    'loss': actual_loss,
    'sharpness': ckpt['sharpness']
  })

traj_alphas = [t['alpha'] for t in trajectory_coords]
traj_betas = [t['beta'] for t in trajectory_coords]
traj_losses = [t['loss'] for t in trajectory_coords]
traj_epochs = [t['epoch'] for t in trajectory_coords]

# ==========================================
# 6. COMPUTE SUMMARY STATISTICS
# ==========================================
sharpness_values = metrics["sharpness"][1:]
max_sharpness = max(sharpness_values)
max_sharpness_epoch = metrics["epoch"][1:][sharpness_values.index(max_sharpness)]
final_sharpness = sharpness_values[-1]
sharpness_reduction = ((max_sharpness - final_sharpness) / max_sharpness) * 100

initial_loss = metrics["loss"][1]
final_loss = metrics["loss"][-1]
loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
loss_absolute_change = initial_loss - final_loss

pca_var1 = variance_explained[0].item() * 100
pca_var2 = variance_explained[1].item() * 100

grid_size = 40
grid_range = max(abs(alpha_min_padded), abs(alpha_max_padded),
                 abs(beta_min_padded), abs(beta_max_padded))

# ==========================================
# 7. CREATE COMBINED FIGURE
# ==========================================
print("Step 5: Creating combined figure...")

fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], width_ratios=[2.5, 1.2, 1.2],
                      hspace=0.3, wspace=0.3)

# Panel A: EoS plot
ax_eos1 = fig.add_subplot(gs[0, :])

# Create plasma colormap for trajectory
epochs_array = np.array(metrics["epoch"][1:])
loss_array = np.array(metrics["loss"][1:])
sharpness_array = np.array(metrics["sharpness"][1:])
colors_plasma = plt.cm.plasma(np.linspace(0, 1, len(epochs_array)))

# Plot loss with plasma gradient (segment by segment)
for i in range(len(epochs_array) - 1):
    ax_eos1.plot(epochs_array[i:i+2], loss_array[i:i+2],
                 color=colors_plasma[i], linewidth=2.5, zorder=10)

ax_eos1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
ax_eos1.set_xlabel('Epochs', fontsize=12)
ax_eos1.tick_params(axis='y', labelsize=11)
ax_eos1.grid(alpha=0.3)
ax_eos1.set_xlim(left=1)
ax_eos1.set_ylim(bottom=0)

# Sharpness on secondary axis
ax_eos2 = ax_eos1.twinx()
color = '#17BECF'
ax_eos2.set_ylabel('Sharpness (Top Eigenvalue)', color=color, fontsize=12, fontweight='bold')
ax_eos2.plot(epochs_array, sharpness_array, color=color, label='Sharpness', linewidth=2.5)
ax_eos2.tick_params(axis='y', labelcolor=color, labelsize=11)
ax_eos2.set_ylim(bottom=0)
ax_eos2.axhline(y=STABILITY_LIMIT, color='#1A237E', linestyle='--', linewidth=2.5,
                label=f'Stability Limit (2/η = {STABILITY_LIMIT:.1f})')

# Shaded instability region
max_sharpness_display = max(sharpness_array) * 1.05
ax_eos2.axhspan(STABILITY_LIMIT, max_sharpness_display,
                alpha=0.15, color='#5C6BC0',
                label='Theoretical Instability Zone', zorder=0)
ax_eos2.set_ylim(bottom=0, top=max_sharpness_display)
ax_eos2.set_xlim(left=1)

# Custom legend combining loss trajectory with sharpness
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color=colors_plasma[0], linewidth=2.5, label='Training Loss (E0→E50)'),
    Line2D([0], [0], color='tab:orange', linewidth=2.5, label='Sharpness'),
    Line2D([0], [0], color='r', linestyle='--', linewidth=2.5, label=f'Stability Limit (2/η = {STABILITY_LIMIT:.1f})'),
    Patch(facecolor='red', alpha=0.15, label='Instability Zone')
]

ax_eos2.legend(handles=legend_elements,
               loc='upper left', bbox_to_anchor=(1.02, 1),
               fontsize=10, frameon=True, fancybox=True)
ax_eos1.text(-0.05, 1.05, 'a', transform=ax_eos1.transAxes,
             fontsize=16, fontweight='bold', va='top')

# Panel B: 3D Landscape
ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
surf = ax_3d.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7,
                          edgecolor='none', antialiased=True, zorder=1)

colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory_coords)))

# Interpolate surface heights at trajectory points
surface_losses = []
for alpha, beta in zip(traj_alphas, traj_betas):
  loss_at_point = griddata(
    (X.flatten(), Y.flatten()),
    Z.flatten(),
    (alpha, beta),
    method='linear'
  )
  surface_losses.append(loss_at_point)

z_offset_percentage = 0.15
traj_z_coords = [surf_loss * (1 + z_offset_percentage) for surf_loss in surface_losses]

# Plot full trajectory
for i in range(len(trajectory_coords) - 1):
  ax_3d.plot(traj_alphas[i:i + 2], traj_betas[i:i + 2],
             [traj_z_coords[i], traj_z_coords[i + 1]],
             color=colors[i], linewidth=2.5, zorder=10)

# Key epochs
key_epochs = [0] + list(range(10, len(trajectory_coords), 2)) + [len(trajectory_coords) - 1]
key_alphas = [traj_alphas[i] for i in key_epochs]
key_betas = [traj_betas[i] for i in key_epochs]
key_z_coords = [traj_z_coords[i] for i in key_epochs]
key_epoch_nums = [traj_epochs[i] for i in key_epochs]

ax_3d.scatter(key_alphas, key_betas, key_z_coords,
              c=key_epoch_nums, cmap='plasma', s=100, marker='o',
              edgecolors='black', linewidths=1.5, zorder=15)

ax_3d.scatter([traj_alphas[0]], [traj_betas[0]], [traj_z_coords[0]],
              color='lime', s=400, marker='o', edgecolors='black',
              linewidths=3, label='Start (E0)', zorder=20)
ax_3d.scatter([traj_alphas[-1]], [traj_betas[-1]], [traj_z_coords[-1]],
              color='red', s=400, marker='*', edgecolors='black',
              linewidths=3, label=f'End (E{EPOCHS})', zorder=20)

# Optimal viewing angle
view_vector_alpha = traj_alphas[-1] - traj_alphas[0]
view_vector_beta = traj_betas[-1] - traj_betas[0]
azim = np.degrees(np.arctan2(view_vector_beta, view_vector_alpha))
azim_optimal = azim % 360 + args.azim_adjust
loss_range = traj_losses[0] - traj_losses[-1]
elev_optimal = (20 if loss_range > 1.0 else 30) + args.elev_adjust

ax_3d.set_axis_off()
ax_3d.set_box_aspect(aspect=None, zoom=args.zoom)
ax_3d.margins(0, 0, 0)
pos = ax_3d.get_position()
ax_3d.set_position([pos.x0 - 0.02, pos.y0, pos.width * 1.2, pos.height * 1.2])
ax_3d.view_init(elev=elev_optimal, azim=azim_optimal - 45)
ax_3d.text2D(-0.1, 1.05, 'b', transform=ax_3d.transAxes,
             fontsize=16, fontweight='bold', va='top')

# Panel C: Contour plot
ax_contour = fig.add_subplot(gs[1, 1])
contour = ax_contour.contourf(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.8)
contour_lines = ax_contour.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)

# Full trajectory line
for i in range(len(trajectory_coords) - 1):
  ax_contour.plot(traj_alphas[i:i + 2], traj_betas[i:i + 2],
                  color=colors[i], linewidth=2.5, zorder=10)

# Only mark key epochs
ax_contour.scatter(key_alphas, key_betas, c=key_epoch_nums, cmap='plasma',
                   s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=15)

# Start/end
ax_contour.scatter(traj_alphas[0], traj_betas[0], color='lime', s=400,
                   marker='o', edgecolors='black', linewidths=3,
                   label='Start (E0)', zorder=20)
ax_contour.scatter(traj_alphas[-1], traj_betas[-1], color='red', s=400,
                   marker='*', edgecolors='black', linewidths=3,
                   label=f'End (E{EPOCHS})', zorder=20)

# Annotate only start, middle, end
mid_idx = len(trajectory_coords) // 2
for idx, label in [(0, 'E0'), (mid_idx, f'E{traj_epochs[mid_idx]}'), (-1, f'E{EPOCHS}')]:
  ax_contour.annotate(label, (traj_alphas[idx], traj_betas[idx]),
                      xytext=(8, 8), textcoords='offset points',
                      fontsize=9, color='darkblue', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax_contour.set_xlabel('PC1 (Primary Direction)', fontsize=10)
ax_contour.set_ylabel('PC2 (Secondary Direction)', fontsize=10)
ax_contour.legend(loc='upper left', fontsize=9)
ax_contour.grid(alpha=0.3)
ax_contour.text(-0.15, 1.05, 'c', transform=ax_contour.transAxes,
                fontsize=16, fontweight='bold', va='top')

# Panel D: Training summary
ax_summary = fig.add_subplot(gs[1, 2])
ax_summary.axis('off')

summary_text = f"""TRAINING CONFIGURATION
Architecture: 3-layer MLP
  {n_channels * 28 * 28}→256→128→{n_classes}
  ReLU activation

Dataset: PathMNIST
  {len(checkpoints)} checkpoints

Optimizer: SGD
  LR: {LEARNING_RATE}
  Batch: {BATCH_SIZE}
  Epochs: {EPOCHS}

EoS METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━
Stability Limit
  2/η = {STABILITY_LIMIT:.2f}

Sharpness
  Peak: {max_sharpness:.2f}
    @ Epoch {max_sharpness_epoch}
  Final: {final_sharpness:.2f}
  Reduction: {sharpness_reduction:.1f}%

Loss
  Initial: {initial_loss:.4f}
  Final: {final_loss:.4f}
  Reduction: {loss_reduction:.1f}%

LANDSCAPE
━━━━━━━━━━━━━━━━━━━━━━━━━━
PCA Variance
  PC1: {pca_var1:.1f}%
  PC2: {pca_var2:.1f}%
  Total: {pca_var1 + pca_var2:.1f}%

Grid: {grid_size}×{grid_size}
Range: [{-grid_range:.1f}, {grid_range:.1f}]
"""

ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                fontsize=10.5, verticalalignment='top', fontfamily='monospace',
                linespacing=1.4)
ax_summary.text(-0.15, 1.05, 'd', transform=ax_summary.transAxes,
                fontsize=16, fontweight='bold', va='top')

# Save
filename_combined = OUTPUT_DIR / f'eos_combined_figure_{CHECKPOINT_INDEX}.png'
plt.savefig(filename_combined, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✓ Combined figure saved as '{filename_combined}'")
print(f"  Full path: {filename_combined.resolve()}")
print("=" * 60)