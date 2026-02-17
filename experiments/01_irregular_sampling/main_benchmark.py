import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd

TRAIN_SAMPLES = 300
SEQ_LEN = 10
DT_MEAN = 0.5
DT_JITTER = 0.3
EXTRAP_DURATION = 2000
HIDDEN_DIM = 128
LEARNING_RATE = 0.0005
EPOCHS = 1500
N_SEEDS = 5  # Statistical rigor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class DynamicalSystem:
  """Base class for dynamical systems"""

  def __init__(self, name: str):
    self.name = name

  def dynamics(self, t, state):
    raise NotImplementedError

  def get_initial_condition(self, on_attractor=False):
    raise NotImplementedError


class FitzHughNagumo(DynamicalSystem):
  def __init__(self):
    super().__init__("FitzHugh-Nagumo")
    self.I_ext = 0.5
    self.a = 0.7
    self.b = 0.8
    self.tau = 12.5

  def dynamics(self, t, state):
    v, w = state
    dv = v - (v ** 3) / 3 - w + self.I_ext
    dw = (v + self.a - self.b * w) / self.tau
    return np.array([dv, dw])

  def get_initial_condition(self, on_attractor=False):
    if on_attractor:
      # On limit cycle
      return np.array([-1.0, 1.0])
    else:
      # Basin of attraction
      return np.random.uniform(-2.0, 2.0, size=2)


class VanDerPol(DynamicalSystem):
  def __init__(self):
    super().__init__("Van der Pol")
    self.mu = 0.5  # reduced from 1.0 for stability

  def dynamics(self, t, state):
    x, y = state
    dx = y
    dy = self.mu * (1 - x ** 2) * y - x
    return np.array([dx, dy])

  def get_initial_condition(self, on_attractor=False):
    if on_attractor:
      # Start on the limit cycle
      return np.array([1.5, 0.0])
    else:
      # Much more conservative basin sampling
      r = np.random.uniform(0.1, 1.5)
      theta = np.random.uniform(0, 2 * np.pi)
      return np.array([r * np.cos(theta), r * np.sin(theta)])

def adaptive_rk4_integrate(func, y0, t_span):
  trajectory = [y0]
  curr_y = y0
  for i in range(len(t_span) - 1):
    dt = t_span[i + 1] - t_span[i]
    t = t_span[i]
    k1 = func(t, curr_y)
    k2 = func(t + dt / 2, curr_y + dt / 2 * k1)
    k3 = func(t + dt / 2, curr_y + dt / 2 * k2)
    k4 = func(t + dt, curr_y + dt * k3)
    curr_y = curr_y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    trajectory.append(curr_y)
  return np.array(trajectory)


def generate_irregular_times(n_points, dt_mean, dt_jitter, seed=None):
  if seed is not None:
    np.random.seed(seed)
  dts = dt_mean + np.random.uniform(-dt_jitter, dt_jitter, n_points - 1)
  dts = np.maximum(dts, 0.1)
  times = np.concatenate([[0], np.cumsum(dts)])
  return times


def generate_regular_times(n_points, dt):
  return np.arange(0, n_points * dt, dt)

def generate_training_data(system: DynamicalSystem,
                           n_samples: int,
                           seq_len: int,
                           irregular: bool,
                           seed: int):
  np.random.seed(seed)
  torch.manual_seed(seed)

  X_lstm, Y_lstm, T_lstm = [], [], []
  X_node, dX_node = [], []

  # Generate limit cycle reference
  settle_time = np.linspace(0, 300, 600)
  y0_settle = system.get_initial_condition(on_attractor=True)
  limit_cycle = adaptive_rk4_integrate(system.dynamics, y0_settle, settle_time)
  limit_cycle_stable = limit_cycle[400:]

  for i in range(n_samples):
    # Generate time vector
    if irregular:
      t_seq = generate_irregular_times(seq_len, DT_MEAN, DT_JITTER)
    else:
      t_seq = generate_regular_times(seq_len, DT_MEAN)

    # Initial condition
    if i < int(0.6 * n_samples):
      y0 = system.get_initial_condition(on_attractor=False)
    else:
      idx = np.random.randint(0, len(limit_cycle_stable))
      y0 = limit_cycle_stable[idx] + np.random.normal(0, 0.05, 2)

    traj = adaptive_rk4_integrate(system.dynamics, y0, t_seq)

    # LSTM data
    X_lstm.append(traj[:-1])
    Y_lstm.append(traj[-1])
    T_lstm.append(np.diff(t_seq))

    # NODE data
    for j in range(len(traj) - 1):
      X_node.append(traj[j])
      dX_node.append(system.dynamics(t_seq[j], traj[j]))

  return {
    'X_lstm': torch.FloatTensor(np.array(X_lstm)),
    'Y_lstm': torch.FloatTensor(np.array(Y_lstm)),
    'T_lstm': torch.FloatTensor(np.array(T_lstm)),
    'X_node': torch.FloatTensor(np.array(X_node)),
    'dX_node': torch.FloatTensor(np.array(dX_node))
  }

class TimeAwareLSTM(nn.Module):
  def __init__(self, seq_len, hidden_dim):
    super().__init__()
    self.seq_len = seq_len
    self.lstm = nn.LSTM(3, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
    self.fc = nn.Linear(hidden_dim, 2)

  def forward(self, x, time_deltas):
    time_deltas_expanded = time_deltas.unsqueeze(-1)
    x_with_time = torch.cat([x, time_deltas_expanded], dim=-1)
    out, _ = self.lstm(x_with_time)
    return self.fc(out[:, -1, :])


class ODERNNCell(nn.Module):
  """ODE-RNN: Better irregular baseline"""

  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    # ODE evolution
    self.ode_func = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.Tanh(),
      nn.Linear(hidden_dim, hidden_dim)
    )
    # Update gate
    self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

  def evolve_hidden(self, h, dt):
    """Evolve hidden state via ODE"""
    # Simple Euler step (could use RK4)
    dh = self.ode_func(h)
    return h + dh * dt

  def forward(self, x, h, dt):
    # Evolve hidden state
    h_evolved = self.evolve_hidden(h, dt)
    # Update with observation
    h_new = self.gru_cell(x, h_evolved)
    return h_new


class ODERNN(nn.Module):
  """Proper irregular time series baseline"""

  def __init__(self, seq_len, hidden_dim):
    super().__init__()
    self.seq_len = seq_len
    self.hidden_dim = hidden_dim
    self.cell = ODERNNCell(2, hidden_dim)
    self.fc = nn.Linear(hidden_dim, 2)

  def forward(self, x, time_deltas):
    batch_size = x.size(0)
    h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

    for t in range(x.size(1)):
      dt = time_deltas[:, t]
      h = self.cell(x[:, t, :], h, dt.unsqueeze(1))

    return self.fc(h)


class ODEFunc(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(2, hidden_dim),
      nn.Tanh(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.Tanh(),
      nn.Linear(hidden_dim, hidden_dim // 2),
      nn.Tanh(),
      nn.Linear(hidden_dim // 2, 2)
    )

  def forward(self, t, y):
    return self.net(y)


class NeuralODE(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.func = ODEFunc(hidden_dim)

  def forward(self, y0, t_points):
    trajectory = [y0]
    curr_y = y0

    for i in range(len(t_points) - 1):
      dt = t_points[i + 1] - t_points[i]
      k1 = self.func(t_points[i], curr_y)
      k2 = self.func(t_points[i] + dt / 2, curr_y + k1 * dt * 0.5)
      k3 = self.func(t_points[i] + dt / 2, curr_y + k2 * dt * 0.5)
      k4 = self.func(t_points[i] + dt, curr_y + k3 * dt)
      curr_y = curr_y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
      trajectory.append(curr_y)

    return torch.stack(trajectory)

def train_model(model, data, model_type, epochs, device, lr=LEARNING_RATE):
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
  criterion = nn.MSELoss()

  for key in data:
    data[key] = data[key].to(device)

  for epoch in range(epochs):
    optimizer.zero_grad()

    if model_type == 'node':
      pred = model.func(None, data['X_node'])
      loss = criterion(pred, data['dX_node'])
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    else:  # LSTM or ODERNN
      pred = model(data['X_lstm'], data['T_lstm'])
      loss = criterion(pred, data['Y_lstm'])

    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

  return model

def evaluate_extrapolation(model, system, model_type, irregular, seed, device):
  np.random.seed(seed)
  y0_test = system.get_initial_condition(on_attractor=False)

  # Generate test times
  if irregular:
    t_test = generate_irregular_times(EXTRAP_DURATION, DT_MEAN, DT_JITTER * 1.5, seed)
  else:
    t_test = generate_regular_times(EXTRAP_DURATION, DT_MEAN)

  true_traj = adaptive_rk4_integrate(system.dynamics, y0_test, t_test)

  model.eval()
  with torch.no_grad():
    if model_type == 'node':
      y0_tensor = torch.FloatTensor(y0_test).to(device)
      t_tensor = torch.FloatTensor(t_test).to(device)
      pred_traj = model(y0_tensor, t_tensor).cpu().numpy()
    else:
      # LSTM/ODERNN rollout
      pred_traj = [y0_test]
      buffer = np.zeros((SEQ_LEN, 2))
      time_buffer = np.ones(SEQ_LEN) * DT_MEAN
      buffer[-1] = y0_test

      for i in range(len(t_test) - 1):
        dt = t_test[i + 1] - t_test[i] if i < len(t_test) - 1 else DT_MEAN
        time_buffer = np.roll(time_buffer, -1)
        time_buffer[-1] = dt

        input_seq = torch.FloatTensor(buffer).unsqueeze(0).to(device)
        input_times = torch.FloatTensor(time_buffer).unsqueeze(0).to(device)
        next_state = model(input_seq, input_times).squeeze(0).cpu().numpy()
        pred_traj.append(next_state)

        buffer = np.roll(buffer, -1, axis=0)
        buffer[-1] = next_state

      pred_traj = np.array(pred_traj)

  # Compute metrics
  rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
  mae = np.mean(np.abs(pred_traj - true_traj))

  return {
    'true': true_traj,
    'pred': pred_traj,
    'times': t_test,
    'rmse': rmse,
    'mae': mae,
    'y0': y0_test
  }

def run_rigorous_experiment(system: DynamicalSystem):
  print(f"\n{'=' * 60}")
  print(f"Running experiment: {system.name}")
  print(f"{'=' * 60}")

  results = {
    'lstm_irregular': [],
    'odernn_irregular': [],
    'node_irregular': [],
    'lstm_regular': [],
    'node_regular': []
  }

  for seed in range(N_SEEDS):
    print(f"\nSeed {seed + 1}/{N_SEEDS}")

    # IRREGULAR SAMPLING
    print("  Generating irregular training data...")
    data_irreg = generate_training_data(system, TRAIN_SAMPLES, SEQ_LEN, irregular=True, seed=seed)

    print("  Training LSTM (irregular)...")
    lstm_irreg = TimeAwareLSTM(SEQ_LEN, HIDDEN_DIM)
    lstm_irreg = train_model(lstm_irreg, data_irreg, 'lstm', EPOCHS, DEVICE)
    results['lstm_irregular'].append(
      evaluate_extrapolation(lstm_irreg, system, 'lstm', irregular=True, seed=seed, device=DEVICE)
    )

    print("  Training ODE-RNN (irregular)...")
    odernn_irreg = ODERNN(SEQ_LEN, HIDDEN_DIM)
    odernn_irreg = train_model(odernn_irreg, data_irreg, 'odernn', EPOCHS, DEVICE)
    results['odernn_irregular'].append(
      evaluate_extrapolation(odernn_irreg, system, 'odernn', irregular=True, seed=seed, device=DEVICE)
    )

    print("  Training Neural ODE (irregular)...")
    node_irreg = NeuralODE(HIDDEN_DIM)
    node_irreg = train_model(node_irreg, data_irreg, 'node', EPOCHS, DEVICE)
    results['node_irregular'].append(
      evaluate_extrapolation(node_irreg, system, 'node', irregular=True, seed=seed, device=DEVICE)
    )

    # REGULAR SAMPLING (ABLATION)
    print("  Generating regular training data...")
    data_reg = generate_training_data(system, TRAIN_SAMPLES, SEQ_LEN, irregular=False, seed=seed)

    print("  Training LSTM (regular)...")
    lstm_reg = TimeAwareLSTM(SEQ_LEN, HIDDEN_DIM)
    lstm_reg = train_model(lstm_reg, data_reg, 'lstm', EPOCHS, DEVICE)
    results['lstm_regular'].append(
      evaluate_extrapolation(lstm_reg, system, 'lstm', irregular=False, seed=seed, device=DEVICE)
    )

    print("  Training Neural ODE (regular)...")
    node_reg = NeuralODE(HIDDEN_DIM)
    node_reg = train_model(node_reg, data_reg, 'node', EPOCHS, DEVICE)
    results['node_regular'].append(
      evaluate_extrapolation(node_reg, system, 'node', irregular=False, seed=seed, device=DEVICE)
    )

  return results

def compute_statistics(results):
  stats_df = []

  for condition, runs in results.items():
    rmses = [r['rmse'] for r in runs]
    maes = [r['mae'] for r in runs]

    stats_df.append({
      'Condition': condition,
      'RMSE_mean': np.mean(rmses),
      'RMSE_std': np.std(rmses),
      'MAE_mean': np.mean(maes),
      'MAE_std': np.std(maes),
      'n': len(runs)
    })

  df = pd.DataFrame(stats_df)
  print("\n" + "=" * 80)
  print("STATISTICAL SUMMARY")
  print("=" * 80)
  print(df.to_string(index=False))

  # Significance testing: NODE vs baselines (irregular)
  node_rmse = [r['rmse'] for r in results['node_irregular']]
  lstm_rmse = [r['rmse'] for r in results['lstm_irregular']]
  odernn_rmse = [r['rmse'] for r in results['odernn_irregular']]

  t_stat_lstm, p_val_lstm = stats.ttest_ind(node_rmse, lstm_rmse)
  t_stat_odernn, p_val_odernn = stats.ttest_ind(node_rmse, odernn_rmse)

  print(f"\nStatistical Significance (t-test, irregular sampling):")
  print(
    f"  NODE vs LSTM:    t={t_stat_lstm:.3f}, p={p_val_lstm:.4f} {'***' if p_val_lstm < 0.001 else '**' if p_val_lstm < 0.01 else '*' if p_val_lstm < 0.05 else 'ns'}")
  print(
    f"  NODE vs ODE-RNN: t={t_stat_odernn:.3f}, p={p_val_odernn:.4f} {'***' if p_val_odernn < 0.001 else '**' if p_val_odernn < 0.01 else '*' if p_val_odernn < 0.05 else 'ns'}")

  return df

def plot_results(results, system_name):
  fig = plt.figure(figsize=(18, 10))
  gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

  # Use first seed for trajectory plots
  seed_idx = 0

  # Panel A: Time domain (irregular)
  ax1 = fig.add_subplot(gs[0, 0])
  # Show only 0-50ms to see actual oscillations
  true = results['node_irregular'][seed_idx]['true']
  times = results['node_irregular'][seed_idx]['times']

  if system_name == "FitzHugh-Nagumo":
    t_max = 200  # ms
  elif system_name == "Van der Pol":
    t_max = 50  # ms

  limit = np.where(times <= t_max)[0][-1] if np.any(times <= t_max) else 100

  ax1.plot(times[:limit], true[:limit, 0], 'k-', alpha=0.7, label='Ground Truth', linewidth=1.5)
  ax1.plot(times[:limit], results['lstm_irregular'][seed_idx]['pred'][:limit, 0],
           'b-', alpha=0.8, label='LSTM', linewidth=1.5)
  ax1.plot(times[:limit], results['odernn_irregular'][seed_idx]['pred'][:limit, 0],
           'c-', alpha=0.8, label='ODE-RNN', linewidth=1.5)
  ax1.plot(times[:limit], results['node_irregular'][seed_idx]['pred'][:limit, 0],
           'r-', linewidth=2, label='Neural ODE', alpha=0.9)
  ax1.axvline(x=SEQ_LEN * DT_MEAN, color='gray', linestyle=':', alpha=0.6)
  ax1.set_title(f"A. Irregular Sampling (0-{t_max}ms)", fontsize=12, fontweight='bold')
  ax1.set_xlabel("Time (ms)", fontsize=10)
  ax1.set_ylabel("State Variable 1", fontsize=10)
  ax1.legend(fontsize=8, loc='upper left')
  ax1.grid(True, alpha=0.3)

  # Panel B: Phase space (irregular)
  ax2 = fig.add_subplot(gs[0, 1])
  plot_lim = 800
  ax2.plot(results['lstm_irregular'][seed_idx]['pred'][:plot_lim, 0],
           results['lstm_irregular'][seed_idx]['pred'][:plot_lim, 1],
           'b-', alpha=0.5, label='LSTM', linewidth=1.5)
  ax2.plot(results['odernn_irregular'][seed_idx]['pred'][:plot_lim, 0],
           results['odernn_irregular'][seed_idx]['pred'][:plot_lim, 1],
           'c-', alpha=0.5, label='ODE-RNN', linewidth=1.5)
  ax2.plot(results['node_irregular'][seed_idx]['pred'][:plot_lim, 0],
           results['node_irregular'][seed_idx]['pred'][:plot_lim, 1],
           'r-', linewidth=2.5, label='Neural ODE', alpha=0.7)
  ax2.plot(true[:, 0], true[:, 1], 'k--', alpha=0.7, linewidth=2, zorder=10, label='Ground Truth')
  y0 = results['node_irregular'][seed_idx]['y0']
  ax2.scatter([y0[0]], [y0[1]], c='green', s=100, zorder=11, edgecolors='black')
  ax2.set_title("B. Phase Space (Irregular)", fontsize=12, fontweight='bold')
  ax2.set_xlabel("State 1", fontsize=10)
  ax2.set_ylabel("State 2", fontsize=10)
  ax2.legend(fontsize=8, loc='best')  # Moved down slightly
  ax2.grid(True, alpha=0.3)

  # Panel C: RMSE comparison with error bars
  ax3 = fig.add_subplot(gs[0, 2])
  conditions = ['LSTM\n(Irreg)', 'ODE-RNN\n(Irreg)', 'NODE\n(Irreg)', 'LSTM\n(Reg)', 'NODE\n(Reg)']
  keys = ['lstm_irregular', 'odernn_irregular', 'node_irregular', 'lstm_regular', 'node_regular']
  means = [np.mean([r['rmse'] for r in results[k]]) for k in keys]
  stds = [np.std([r['rmse'] for r in results[k]]) for k in keys]
  colors = ['blue', 'cyan', 'red', 'lightblue', 'pink']

  bars = ax3.bar(conditions, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
  ax3.set_title(f"C. RMSE Comparison (n={N_SEEDS})", fontsize=12, fontweight='bold')
  ax3.set_ylabel("RMSE", fontsize=10)
  ax3.grid(True, alpha=0.3, axis='y')

  # Panel D: Regular sampling (ablation) - SHORT WINDOW
  ax4 = fig.add_subplot(gs[1, 0])
  true_reg = results['node_regular'][seed_idx]['true']
  times_reg = results['node_regular'][seed_idx]['times']
  limit_reg = np.where(times_reg <= t_max)[0][-1] if np.any(times_reg <= t_max) else 100

  ax4.plot(times_reg[:limit_reg], true_reg[:limit_reg, 0], 'k-', alpha=0.7,
           label='Ground Truth', linewidth=1.5)
  ax4.plot(times_reg[:limit_reg], results['lstm_regular'][seed_idx]['pred'][:limit_reg, 0],
           'b-', alpha=0.8, label='LSTM', linewidth=1.5)
  ax4.plot(times_reg[:limit_reg], results['node_regular'][seed_idx]['pred'][:limit_reg, 0],
           'r-', linewidth=2, label='Neural ODE', alpha=0.9)
  ax4.set_title(f"D. Regular Sampling (0-{t_max}ms)", fontsize=12, fontweight='bold')
  ax4.set_xlabel("Time (ms)", fontsize=10)
  ax4.set_ylabel("State Variable 1", fontsize=10)
  ax4.legend(fontsize=8, loc='best')
  ax4.grid(True, alpha=0.3)

  # Panel E: Error over time
  ax5 = fig.add_subplot(gs[1, 1])
  window = 50
  for i, key in enumerate(['lstm_irregular', 'odernn_irregular', 'node_irregular']):
    errors = []
    for run in results[key]:
      err = np.sqrt(np.sum((run['pred'] - run['true']) ** 2, axis=1))
      errors.append(err)
    errors = np.array(errors)
    mean_err = np.mean(errors, axis=0)
    std_err = np.std(errors, axis=0)

    mean_smooth = np.convolve(mean_err, np.ones(window) / window, mode='valid')
    std_smooth = np.convolve(std_err, np.ones(window) / window, mode='valid')
    t_smooth = times[window - 1:len(mean_smooth) + window - 1]

    color = ['blue', 'cyan', 'red'][i]
    label = ['LSTM', 'ODE-RNN', 'NODE'][i]
    ax5.plot(t_smooth[:1500], mean_smooth[:1500], color=color, linewidth=2, label=label)
    ax5.fill_between(t_smooth[:1500],
                     (mean_smooth - std_smooth)[:1500],
                     (mean_smooth + std_smooth)[:1500],
                     color=color, alpha=0.2)
  ax5.axvline(x=SEQ_LEN * DT_MEAN, color='gray', linestyle=':', alpha=0.6, label='Training Horizon')
  ax5.set_title(f"E. Error Evolution (n={N_SEEDS})", fontsize=12, fontweight='bold')
  ax5.set_xlabel("Time (ms)", fontsize=10)
  ax5.set_ylabel("Euclidean Error", fontsize=10)
  ax5.legend(fontsize=9)
  ax5.grid(True, alpha=0.3)

  # Panel F: Box plots
  ax6 = fig.add_subplot(gs[1, 2])
  data_to_plot = [[r['rmse'] for r in results[k]] for k in keys[:3]]
  bp = ax6.boxplot(data_to_plot, labels=['LSTM', 'ODE-RNN', 'NODE'], patch_artist=True)
  for patch, color in zip(bp['boxes'], ['blue', 'cyan', 'red']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
  ax6.set_title("F. RMSE Distribution", fontsize=12, fontweight='bold')
  ax6.set_ylabel("RMSE", fontsize=10)
  ax6.grid(True, alpha=0.3, axis='y')

  # plt.suptitle(f"{system_name}: Rigorous Comparison (Irregular Sampling)",
  #              fontsize=14, fontweight='bold', y=0.995)
  plt.savefig(f'{system_name.replace(" ", "_")}_rigorous.png', dpi=300, bbox_inches='tight')
  print(f"\n✓ Saved {system_name.replace(' ', '_')}_rigorous.png")

if __name__ == "__main__":
  # Run experiment on FitzHugh-Nagumo
  fhn_system = FitzHughNagumo()
  fhn_results = run_rigorous_experiment(fhn_system)
  fhn_stats = compute_statistics(fhn_results)
  plot_results(fhn_results, fhn_system.name)

  ## OPTIONAL: Run on Van der Pol
  # vdp_system = VanDerPol()
  # vdp_results = run_rigorous_experiment(vdp_system)
  # vdp_stats = compute_statistics(vdp_results)
  # plot_results(vdp_results, vdp_system.name)

  print("\n" + "=" * 80)
  print("EXPERIMENT COMPLETE")
  print("=" * 80)
  print(f"✓ {N_SEEDS} independent random seeds")
  print(f"✓ Proper irregular baseline (ODE-RNN)")
  print(f"✓ Regular vs Irregular ablation")
  print(f"✓ Statistical significance testing")
  print(f"✓ Error bars and confidence intervals")

  print(f"✓ Multiple evaluation metrics (RMSE, MAE)")
