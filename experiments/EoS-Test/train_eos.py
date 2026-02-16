import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pickle
from pathlib import Path
import argparse
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader

# ==========================================
# 0. COMMAND LINE ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description='Train with EoS tracking')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed for reproducibility')
parser.add_argument('--base-index', type=int, default=None,  # ADD THIS
                    help='Force specific checkpoint base index (useful for multi-seed runs)')
args = parser.parse_args()

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
CHECKPOINTS_DIR = Path('./checkpoints')
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def get_next_index_from_checkpoints():
  """Find next index by checking checkpoints folder."""
  existing = list(CHECKPOINTS_DIR.glob('checkpoint_*.pkl'))
  if not existing:
    return 0

  indices = []
  for f in existing:
    try:
      # Extract number from checkpoint_XX_seedY.pkl or checkpoint_XX.pkl
      name = f.stem.replace('checkpoint_', '').split('_seed')[0]
      idx = int(name)
      indices.append(idx)
    except:
      continue

  return max(indices) + 1 if indices else 0


# Use explicit base index if provided, otherwise auto-increment
if args.base_index is not None:
  CHECKPOINT_INDEX = args.base_index
  print(f"Using forced checkpoint index: {CHECKPOINT_INDEX}")
else:
  CHECKPOINT_INDEX = get_next_index_from_checkpoints()
  print(f"Next available index: {CHECKPOINT_INDEX}")

seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
CHECKPOINT_FILE = CHECKPOINTS_DIR / f'checkpoint_{CHECKPOINT_INDEX}{seed_suffix}.pkl'

# Set seed if provided
if args.seed is not None:
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
  import numpy as np

  np.random.seed(args.seed)
  print(f"Random seed: {args.seed}")

DATA_DIR = Path('/home/zzai2932/Data/MedMNIST')
DATA_FLAG = 'pathmnist'

BATCH_SIZE = 128
LEARNING_RATE = 0.15
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FREQUENCY = 1
STABILITY_LIMIT = 2.0 / LEARNING_RATE

print(f"Next available index: {CHECKPOINT_INDEX}")
print(f"Checkpoint file: {CHECKPOINT_FILE}")
print(f"Running on: {DEVICE}")
print(f"Dataset: {DATA_FLAG}")
print(f"Data directory: {DATA_DIR.resolve()}")
print(f"Stability Threshold (2/eta): {STABILITY_LIMIT:.2f}")

# ==========================================
# 2. DATASET & MODEL
# ==========================================
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

DATA_DIR.mkdir(parents=True, exist_ok=True)

info = INFO[DATA_FLAG]
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
trainset = DataClass(
  split='train',
  root=str(DATA_DIR),
  transform=transform,
  download=False
)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

eigen_loader = DataLoader(trainset, batch_size=256, shuffle=False)
eigen_data, eigen_target = next(iter(eigen_loader))
eigen_target = eigen_target.squeeze().long()
eigen_data, eigen_target = eigen_data.to(DEVICE), eigen_target.to(DEVICE)


# ==========================================
# 3. MODEL
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.0)


# ==========================================
# 4. SHARPNESS COMPUTATION
# ==========================================
def compute_top_eigenvalue(model, data, target, criterion, max_iter=50, tol=1e-4):
  """Computes the top eigenvalue of the Hessian using Power Iteration."""
  params = [p for p in model.parameters() if p.requires_grad]
  v = [torch.randn_like(p).to(p.device) for p in params]
  v_norm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
  v = [vi / v_norm for vi in v]
  eigenvalue = 0.0

  for iteration in range(max_iter):
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_dot_v = sum((g * vi).sum() for g, vi in zip(grads, v))
    Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=False)
    Hv_norm = torch.sqrt(sum(torch.sum(hv ** 2) for hv in Hv))
    v = [hv.detach() / (Hv_norm + 1e-10) for hv in Hv]
    new_eigenvalue = Hv_norm.item()

    if iteration > 0 and abs(new_eigenvalue - eigenvalue) < tol:
      break
    eigenvalue = new_eigenvalue

  return eigenvalue


# ==========================================
# 5. TRAINING LOOP
# ==========================================
metrics = {"loss": [], "sharpness": [], "epoch": []}
checkpoints = []

print("Measuring initial sharpness...")
initial_sharpness = compute_top_eigenvalue(model, eigen_data, eigen_target, criterion)
metrics["sharpness"].append(initial_sharpness)
metrics["loss"].append(0.0)
metrics["epoch"].append(0)

checkpoints.append({
  'epoch': 0,
  'params': [p.clone().detach().cpu() for p in model.parameters()],
  'loss': 0.0,
  'sharpness': initial_sharpness
})

print(f"Initial Sharpness: {initial_sharpness:.4f}")
print("\nStarting training...")

for epoch in range(EPOCHS):
  model.train()
  running_loss = 0.0

  for batch_idx, (data, target) in enumerate(trainloader):
    data, target = data.to(DEVICE), target.to(DEVICE)
    target = target.squeeze().long()

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  avg_loss = running_loss / len(trainloader)

  model.train()
  sharpness = compute_top_eigenvalue(model, eigen_data, eigen_target, criterion)

  metrics["loss"].append(avg_loss)
  metrics["sharpness"].append(sharpness)
  metrics["epoch"].append(epoch + 1)

  if (epoch + 1) % CHECKPOINT_FREQUENCY == 0 or epoch == EPOCHS - 1:
    checkpoints.append({
      'epoch': epoch + 1,
      'params': [p.clone().detach().cpu() for p in model.parameters()],
      'loss': avg_loss,
      'sharpness': sharpness
    })
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} | Sharpness: {sharpness:.2f} ✓ Checkpoint saved")
  else:
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} | Sharpness: {sharpness:.2f}")

# ==========================================
# 6. SAVE TRAINING RESULTS
# ==========================================
print("\nSaving training results...")

training_data = {
  'checkpoints': checkpoints,
  'metrics': metrics,
  'model_state_dict': model.state_dict(),
  'eigen_data': eigen_data.cpu(),
  'eigen_target': eigen_target.cpu(),
  'hyperparameters': {
    'learning_rate': LEARNING_RATE,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'stability_limit': STABILITY_LIMIT,
    'n_channels': n_channels,
    'n_classes': n_classes,
    'seed': args.seed
  }
}

with open(CHECKPOINT_FILE, 'wb') as f:
  pickle.dump(training_data, f)

print(f"✓ Saved to {CHECKPOINT_FILE}")
print(f"File size: {Path(CHECKPOINT_FILE).stat().st_size / 1024 / 1024:.2f} MB")
print("\nTraining complete! Run visualize.py or visualize_multi.py to generate figures.")
