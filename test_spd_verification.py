"""
Test script for Neural KF v23 - Simple Baseline with Diagnostics

Key question: Can the network learn velocity dynamics from range measurements alone?
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from nn_train import train_v23, NeuralKFv23, SimpleDynamicsNet, FixedCovPredictor, StateScaler, UKFCore, analyze_gradients

torch.set_default_dtype(torch.float32)
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# CONFIG
# =============================================================================
nx, nz = 4, 2
dt = 0.5
RADAR_BASELINE = 150.0

F_true = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float32)
sigma_u = 0.01
Q_true = np.array([[dt**3/3, dt**2/2, 0, 0], [dt**2/2, dt, 0, 0],
                   [0, 0, dt**3/3, dt**2/2], [0, 0, dt**2/2, dt]], dtype=np.float32) * sigma_u**2
sigma_r = 1.0
R_true = np.eye(nz, dtype=np.float32) * sigma_r**2
x0_mean = np.array([100.0, 1.0, 50.0, 0.5], dtype=np.float32)
P0 = np.diag([10.0, 1.0, 10.0, 1.0]).astype(np.float32)

n_train, n_test, n_timesteps = 2000, 100, 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def h_numpy(x):
    r1 = np.sqrt(x[0]**2 + x[2]**2 + 1e-12)
    r2 = np.sqrt((x[0] - RADAR_BASELINE)**2 + x[2]**2 + 1e-12)
    return np.array([r1, r2], dtype=np.float32)

def h_torch(x):
    r1 = torch.sqrt(x[:, 0]**2 + x[:, 2]**2 + 1e-12)
    r2 = torch.sqrt((x[:, 0] - RADAR_BASELINE)**2 + x[:, 2]**2 + 1e-12)
    return torch.stack([r1, r2], dim=-1)

def generate_data(n_seq, T):
    states = np.zeros((n_seq, T, nx), dtype=np.float32)
    measurements = np.zeros((n_seq, T, nz), dtype=np.float32)
    L_Q = np.linalg.cholesky(Q_true + 1e-8 * np.eye(nx))
    L_R = np.linalg.cholesky(R_true)
    L_P0 = np.linalg.cholesky(P0)
    for i in range(n_seq):
        x = x0_mean + L_P0 @ np.random.randn(nx).astype(np.float32)
        for t in range(T):
            states[i, t] = x
            measurements[i, t] = h_numpy(x) + L_R @ np.random.randn(nz).astype(np.float32)
            x = F_true @ x + L_Q @ np.random.randn(nx).astype(np.float32)
    return states, measurements

class UKFBaseline:
    def __init__(self, F, Q, R, h_fn, alpha=0.001, beta=2.0, kappa=0.0):
        self.F, self.Q, self.R, self.h = F, Q, R, h_fn
        self.nx, self.nz = F.shape[0], R.shape[0]
        lmbda = alpha**2 * (self.nx + kappa) - self.nx
        self.gamma = np.sqrt(self.nx + lmbda)
        n = 2 * self.nx + 1
        self.Wm = np.zeros(n); self.Wc = np.zeros(n)
        self.Wm[0] = lmbda / (self.nx + lmbda)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        self.Wm[1:] = 0.5 / (self.nx + lmbda)
        self.Wc[1:] = 0.5 / (self.nx + lmbda)
        self.n_sigma = n
    
    def _ensure_spd(self, P):
        P = 0.5 * (P + P.T)
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-6)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def run(self, z_seq, x0, P0):
        T = z_seq.shape[0]
        xs, Ps = np.zeros((T, self.nx)), np.zeros((T, self.nx, self.nx))
        x, P = x0.copy(), P0.copy()
        for t in range(T):
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            P_pred = self._ensure_spd(P_pred)
            try:
                L = np.linalg.cholesky(P_pred + 1e-6 * np.eye(self.nx))
            except:
                P_pred = self._ensure_spd(P_pred + 1e-4 * np.eye(self.nx))
                L = np.linalg.cholesky(P_pred)
            sigma = np.zeros((self.n_sigma, self.nx))
            sigma[0] = x_pred
            for i in range(self.nx):
                sigma[i+1] = x_pred + self.gamma * L[:, i]
                sigma[self.nx+i+1] = x_pred - self.gamma * L[:, i]
            z_sigma = np.array([self.h(s) for s in sigma])
            z_hat = self.Wm @ z_sigma
            S, C = self.R.copy(), np.zeros((self.nx, self.nz))
            for i in range(self.n_sigma):
                dz, dx = z_sigma[i] - z_hat, sigma[i] - x_pred
                S += self.Wc[i] * np.outer(dz, dz)
                C += self.Wc[i] * np.outer(dx, dz)
            K = C @ np.linalg.inv(S)
            x = x_pred + K @ (z_seq[t] - z_hat)
            P = P_pred - K @ S @ K.T
            P = self._ensure_spd(P)
            xs[t], Ps[t] = x, P
        return xs, Ps

# =============================================================================
# GRADIENT ANALYSIS (Before training)
# =============================================================================
print("\n" + "="*60)
print("GRADIENT ANALYSIS - Understanding what's learnable")
print("="*60)

gt_train, z_train = generate_data(n_train, n_timesteps)
gt_test, z_test = generate_data(n_test, n_timesteps)

z_train_t = torch.as_tensor(z_train, dtype=torch.float32, device=device)

# Initialize fresh networks
scaler = StateScaler(100.0, 1.0, device)
mean_net = SimpleDynamicsNet(nx).to(device)
cov_net = FixedCovPredictor(nx).to(device)
ukf = UKFCore(nx, nz, R_true, device)

# Analyze gradients
grads = analyze_gradients(mean_net, cov_net, scaler, ukf, z_train_t[:100], x0_mean, P0, h_torch, device)

print("\nGradient magnitudes (larger = more learnable):")
for k, v in grads.items():
    print(f"  {k}: {v:.6f}")

print("\nInterpretation:")
print("  - If d_NLL is much smaller than d_pred_x/y, velocity has weak gradient signal")
print("  - The network may not be able to learn velocity dynamics from NLL alone")

# =============================================================================
# MAIN TRAINING
# =============================================================================
output_dir = "./test_output_v23"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("TRAINING v23 - SIMPLE BASELINE")
print("="*60)

mean_net, cov_net, scaler, history = train_v23(
    z_train=z_train, R=R_true, h_fn=h_torch,
    x0=x0_mean, P0=P0, nx=nx, nz=nz,
    pos_scale=100.0, vel_scale=1.0,
    epochs=400, lr=1e-3, device=str(device),
    checkpoint_path=os.path.join(output_dir, 'neural_kf_v23.pth'),
    verbose=True
)

# Plot training
plt.figure(figsize=(10, 5))
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (NLL)')
plt.title(f"Training Loss (best @ epoch {history['best_epoch']})")
plt.axvline(200, color='r', ls='--', label='Phase 2 (unfreeze cov)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150)
plt.close()

# =============================================================================
# TESTING
# =============================================================================
print("\n" + "="*60)
print("TESTING")
print("="*60)

neural_kf = NeuralKFv23(mean_net, cov_net, scaler, R_true, device)
ukf_baseline = UKFBaseline(F_true, Q_true, R_true, h_numpy)

n_eval = 50
neural_rmse_all, ukf_rmse_all = [], []
neural_states_all, ukf_states_all, gt_all = [], [], []
neural_covs_all, ukf_covs_all = [], []

for seq_idx in range(n_eval):
    n_states, n_covs = neural_kf.run(z_test[seq_idx], x0_mean, P0, h_torch)
    u_states, u_covs = ukf_baseline.run(z_test[seq_idx], x0_mean, P0)
    g = gt_test[seq_idx]
    
    neural_states_all.append(n_states[0])
    ukf_states_all.append(u_states)
    gt_all.append(g)
    neural_covs_all.append(n_covs[0])
    ukf_covs_all.append(u_covs)
    
    n_rmse = np.sqrt(np.mean((n_states[0][:, [0,2]] - g[:, [0,2]])**2, axis=1))
    u_rmse = np.sqrt(np.mean((u_states[:, [0,2]] - g[:, [0,2]])**2, axis=1))
    
    neural_rmse_all.append(np.mean(n_rmse))
    ukf_rmse_all.append(np.mean(u_rmse))

neural_states_all = np.array(neural_states_all)
ukf_states_all = np.array(ukf_states_all)
gt_all = np.array(gt_all)
neural_covs_all = np.array(neural_covs_all)
ukf_covs_all = np.array(ukf_covs_all)

print(f"Average Position RMSE over {n_eval} sequences:")
print(f"  Neural KF v23: {np.mean(neural_rmse_all):.3f} +/- {np.std(neural_rmse_all):.3f}")
print(f"  UKF (oracle):  {np.mean(ukf_rmse_all):.3f} +/- {np.std(ukf_rmse_all):.3f}")

neural_vel_rmse = np.sqrt(np.mean((neural_states_all[:, :, [1,3]] - gt_all[:, :, [1,3]])**2))
ukf_vel_rmse = np.sqrt(np.mean((ukf_states_all[:, :, [1,3]] - gt_all[:, :, [1,3]])**2))
print(f"\nAverage Velocity RMSE:")
print(f"  Neural KF v23: {neural_vel_rmse:.3f}")
print(f"  UKF (oracle):  {ukf_vel_rmse:.3f}")

# NEES
neural_nees_all = []
ukf_nees_all = []
for seq_idx in range(n_eval):
    for t in range(n_timesteps):
        err_n = neural_states_all[seq_idx, t] - gt_all[seq_idx, t]
        err_u = ukf_states_all[seq_idx, t] - gt_all[seq_idx, t]
        P_n = neural_covs_all[seq_idx, t]
        P_u = ukf_covs_all[seq_idx, t]
        try:
            neural_nees_all.append(err_n @ np.linalg.inv(P_n + 1e-6*np.eye(nx)) @ err_n)
            ukf_nees_all.append(err_u @ np.linalg.inv(P_u + 1e-6*np.eye(nx)) @ err_u)
        except:
            pass

print(f"\nNEES (should be ~{nx}):")
print(f"  Neural KF v23: {np.mean(neural_nees_all):.2f}")
print(f"  UKF (oracle):  {np.mean(ukf_nees_all):.2f}")

# =============================================================================
# DYNAMICS ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("LEARNED DYNAMICS ANALYSIS")
print("="*60)

mean_net.eval()
with torch.no_grad():
    # Test various states
    test_states_norm = torch.tensor([
        [1.0, 0.00, 0.5, 0.00],   # stationary
        [1.0, 0.01, 0.5, 0.005],  # vx=1, vy=0.5
        [1.0, 0.02, 0.5, 0.01],   # vx=2, vy=1
        [1.0, -0.01, 0.5, -0.005],# backward
    ], device=device, dtype=torch.float32)
    
    deltas = mean_net.get_delta(test_states_norm).cpu().numpy()
    
    print("\nLearned deltas (normalized space):")
    print("-" * 60)
    labels = ['stationary', 'vx=1,vy=0.5', 'vx=2,vy=1', 'backward']
    for i, lbl in enumerate(labels):
        d = deltas[i]
        # Expected for CV: dx = vx * dt / pos_scale = vx_norm * vel_scale * dt / pos_scale
        # With vel_scale=1, pos_scale=100, dt=0.5: dx = vx_norm * 0.005
        vx_norm = test_states_norm[i, 1].item()
        expected_dx = vx_norm * 0.005
        print(f"  {lbl:15s}: Δx={d[0]:.5f} (expect {expected_dx:.5f}), Δvx={d[1]:.5f}")

    # Sweep vx to see if delta depends on it
    vx_range = np.linspace(-0.02, 0.02, 21)
    dx_values = []
    for vx in vx_range:
        x_test = torch.tensor([[1.0, vx, 0.5, 0.0]], device=device, dtype=torch.float32)
        delta = mean_net.get_delta(x_test)[0, 0].cpu().item()
        dx_values.append(delta)
    
    corr = np.corrcoef(vx_range, dx_values)[0, 1]
    slope, intercept = np.polyfit(vx_range, dx_values, 1)
    
    print(f"\nΔx vs vx analysis:")
    print(f"  Correlation: {corr:.3f} (should be ~1.0 for CV)")
    print(f"  Slope: {slope:.5f} (should be ~0.005 for CV)")
    print(f"  Intercept: {intercept:.5f} (should be ~0)")

# =============================================================================
# PLOTS
# =============================================================================
t_plot = np.arange(n_timesteps)

# State comparison
test_idx = 0
gt = gt_all[test_idx]
neural_states = neural_states_all[test_idx]
ukf_states = ukf_states_all[test_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = ['x (m)', 'vx (m/s)', 'y (m)', 'vy (m/s)']
for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
    ax.plot(t_plot, gt[:, i], 'k-', lw=2, label='Ground Truth')
    ax.plot(t_plot, neural_states[:, i], 'b-', alpha=0.8, label='Neural KF v23')
    ax.plot(t_plot, ukf_states[:, i], 'r--', alpha=0.8, label='UKF (oracle)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel(lbl)
    ax.legend()
    ax.grid(True)
plt.suptitle('State Estimation', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_comparison.png'), dpi=150)
plt.close()

# Δx vs vx plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(vx_range, dx_values, 'b.-', lw=2, ms=8, label='Learned')
ax.plot(vx_range, vx_range * 0.005, 'r--', lw=2, label='Expected (CV)')
ax.set_xlabel('vx (normalized)')
ax.set_ylabel('Δx (normalized)')
ax.set_title(f'Does Δx depend on vx?\nCorrelation: {corr:.3f}, Slope: {slope:.5f}')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'delta_vs_velocity.png'), dpi=150)
plt.close()

# Variance comparison
neural_vars = np.diagonal(neural_covs_all, axis1=2, axis2=3)
ukf_vars = np.diagonal(ukf_covs_all, axis1=2, axis2=3)
neural_vars_mean = np.mean(neural_vars, axis=0)
ukf_vars_mean = np.mean(ukf_vars, axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
state_labels = ['σ_x', 'σ_vx', 'σ_y', 'σ_vy']
for i, (ax, lbl) in enumerate(zip(axes.flat, state_labels)):
    ax.semilogy(t_plot, neural_vars_mean[:, i], 'b-', lw=2, label='Neural KF v23')
    ax.semilogy(t_plot, ukf_vars_mean[:, i], 'r--', lw=2, label='UKF (oracle)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Variance')
    ax.set_title(f'{lbl} Variance')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'variance_comparison.png'), dpi=150)
plt.close()

print(f"\nAll plots saved to: {output_dir}")

# =============================================================================
# FINAL DIAGNOSIS
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if abs(corr) < 0.5:
    print("❌ The network did NOT learn that Δx depends on vx")
    print("   This confirms that NLL loss alone cannot teach velocity dynamics")
    print("   from range measurements - the gradient signal is too weak.")
    print("\n   POSSIBLE SOLUTIONS:")
    print("   1. Use supervised pre-training with known F")
    print("   2. Add auxiliary loss that directly rewards velocity prediction")
    print("   3. Use different measurement model with better velocity observability")
    print("   4. Accept that this is a fundamental limitation of the approach")
else:
    print("✓ The network learned some velocity dependence!")
    print(f"  Correlation: {corr:.3f}, Slope: {slope:.5f} (expected 0.005)")