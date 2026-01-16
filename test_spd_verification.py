# test_spd_verification.py
"""Test v10 - Residual architecture."""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from nn_train import train_v10, NeuralKFv10

torch.set_default_dtype(torch.float32)
np.random.seed(42)
torch.manual_seed(42)

# Config
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

n_train, n_test, n_timesteps = 2000, 100, 100
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

print("Generating data...")
gt_train, z_train = generate_data(n_train, n_timesteps)
gt_test, z_test = generate_data(n_test, n_timesteps)
print(f"Train: {z_train.shape}, Test: {z_test.shape}")

output_dir = "./test_output_v10"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("TRAINING v10 - RESIDUAL ARCHITECTURE")
print("="*60)

mean_net, cov_net, scaler, history = train_v10(
    z_train=z_train, R=R_true, h_fn=h_torch,
    x0=x0_mean, P0=P0, nx=nx, nz=nz,
    dt=dt, pos_scale=100.0, vel_scale=5.0,
    epochs=400, lr=1e-3, device=str(device),
    checkpoint_path=os.path.join(output_dir, 'neural_kf_v10.pth'),
    tbptt_len=5, verbose=True
)

plt.figure(figsize=(10, 4))
plt.plot(history['loss'])
plt.xlabel('Epoch'); plt.ylabel('Loss (NLL)')
plt.title(f"Training Loss (best @ epoch {history['best_epoch']})")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150)
plt.close()

# UKF Baseline with robust covariance handling
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
        """Ensure matrix is symmetric positive definite."""
        P = 0.5 * (P + P.T)  # Symmetrize
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-6)  # Clamp eigenvalues
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
            P = self._ensure_spd(P)  # Ensure SPD after update
            xs[t], Ps[t] = x, P
        return xs, Ps

print("\n" + "="*60)
print("TESTING")
print("="*60)

neural_kf = NeuralKFv10(mean_net, cov_net, scaler, R_true, device)
ukf = UKFBaseline(F_true, Q_true, R_true, h_numpy)

test_idx = 0
neural_states, _ = neural_kf.run(z_test[test_idx], x0_mean, P0, h_torch)
neural_states = neural_states[0]
ukf_states, _ = ukf.run(z_test[test_idx], x0_mean, P0)
gt = gt_test[test_idx]

rmse_neural = np.sqrt(np.mean((neural_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))
rmse_ukf = np.sqrt(np.mean((ukf_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))

print(f"Single sequence RMSE:")
print(f"  Neural KF v10: mean={rmse_neural.mean():.3f}")
print(f"  UKF:           mean={rmse_ukf.mean():.3f}")

# Plots
t_plot = np.arange(n_timesteps)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = ['x (m)', 'vx (m/s)', 'y (m)', 'vy (m/s)']
for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
    ax.plot(t_plot, gt[:, i], 'k-', lw=2, label='Ground Truth')
    ax.plot(t_plot, neural_states[:, i], 'b-', alpha=0.8, label='Neural KF v10')
    ax.plot(t_plot, ukf_states[:, i], 'r--', alpha=0.8, label='UKF')
    ax.set_xlabel('Time Step'); ax.set_ylabel(lbl)
    ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_comparison.png'), dpi=150)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(gt[:, 0], gt[:, 2], 'k-', lw=2, label='Ground Truth')
plt.plot(neural_states[:, 0], neural_states[:, 2], 'b-', alpha=0.8, label='Neural KF v10')
plt.plot(ukf_states[:, 0], ukf_states[:, 2], 'r--', alpha=0.8, label='UKF')
plt.plot(0, 0, 'g^', ms=10); plt.plot(RADAR_BASELINE, 0, 'm^', ms=10)
plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.legend(); plt.grid(True); plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'xy_trajectory.png'), dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(t_plot, rmse_neural, 'b-', lw=2, label='Neural KF v10')
plt.plot(t_plot, rmse_ukf, 'r--', lw=2, label='UKF')
plt.xlabel('Time Step'); plt.ylabel('Position RMSE (m)')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'), dpi=150)
plt.close()

# Multi-sequence
print("\n" + "="*60)
print("MULTI-SEQUENCE EVALUATION")
print("="*60)

neural_rmse_all, ukf_rmse_all = [], []
neural_rmse_per_t = np.zeros((50, n_timesteps))
ukf_rmse_per_t = np.zeros((50, n_timesteps))

for seq_idx in range(50):
    n_states, _ = neural_kf.run(z_test[seq_idx], x0_mean, P0, h_torch)
    n_states = n_states[0]
    u_states, _ = ukf.run(z_test[seq_idx], x0_mean, P0)
    g = gt_test[seq_idx]
    
    # Per-timestep RMSE for this sequence
    neural_rmse_t = np.sqrt(np.mean((n_states[:, [0,2]] - g[:, [0,2]])**2, axis=1))
    ukf_rmse_t = np.sqrt(np.mean((u_states[:, [0,2]] - g[:, [0,2]])**2, axis=1))
    
    neural_rmse_per_t[seq_idx] = neural_rmse_t
    ukf_rmse_per_t[seq_idx] = ukf_rmse_t
    
    neural_rmse_all.append(np.mean(neural_rmse_t))
    ukf_rmse_all.append(np.mean(ukf_rmse_t))

print(f"Average RMSE over 50 sequences:")
print(f"  Neural KF v10: {np.mean(neural_rmse_all):.3f} +/- {np.std(neural_rmse_all):.3f}")
print(f"  UKF:           {np.mean(ukf_rmse_all):.3f} +/- {np.std(ukf_rmse_all):.3f}")

# Box plot
plt.figure(figsize=(8, 6))
plt.boxplot([neural_rmse_all, ukf_rmse_all], labels=['Neural KF v10', 'UKF'])
plt.ylabel('Position RMSE (m)')
plt.grid(True, axis='y')
plt.savefig(os.path.join(output_dir, 'rmse_boxplot.png'), dpi=150)
plt.close()

# RMSE vs Time across all sequences
plt.figure(figsize=(12, 5))

neural_mean = np.mean(neural_rmse_per_t, axis=0)
neural_std = np.std(neural_rmse_per_t, axis=0)
ukf_mean = np.mean(ukf_rmse_per_t, axis=0)
ukf_std = np.std(ukf_rmse_per_t, axis=0)

plt.subplot(1, 2, 1)
plt.plot(t_plot, neural_mean, 'b-', lw=2, label='Neural KF v10')
plt.fill_between(t_plot, neural_mean - neural_std, neural_mean + neural_std, alpha=0.3, color='blue')
plt.plot(t_plot, ukf_mean, 'r--', lw=2, label='UKF')
plt.fill_between(t_plot, ukf_mean - ukf_std, ukf_mean + ukf_std, alpha=0.3, color='red')
plt.xlabel('Time Step')
plt.ylabel('Position RMSE (m)')
plt.title('Mean RMSE +/- Std over 50 Sequences')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
neural_median = np.median(neural_rmse_per_t, axis=0)
neural_p25 = np.percentile(neural_rmse_per_t, 25, axis=0)
neural_p75 = np.percentile(neural_rmse_per_t, 75, axis=0)
ukf_median = np.median(ukf_rmse_per_t, axis=0)
ukf_p25 = np.percentile(ukf_rmse_per_t, 25, axis=0)
ukf_p75 = np.percentile(ukf_rmse_per_t, 75, axis=0)

plt.plot(t_plot, neural_median, 'b-', lw=2, label='Neural KF v10')
plt.fill_between(t_plot, neural_p25, neural_p75, alpha=0.3, color='blue')
plt.plot(t_plot, ukf_median, 'r--', lw=2, label='UKF')
plt.fill_between(t_plot, ukf_p25, ukf_p75, alpha=0.3, color='red')
plt.xlabel('Time Step')
plt.ylabel('Position RMSE (m)')
plt.title('Median RMSE (25-75 percentile) over 50 Sequences')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rmse_vs_time_all_sequences.png'), dpi=150)
plt.close()

print(f"\nPer-timestep stats (t=0, t=10, t=49):")
for t_idx in [0, 10, 49]:
    print(f"  t={t_idx}: Neural={neural_mean[t_idx]:.2f}+/-{neural_std[t_idx]:.2f}, UKF={ukf_mean[t_idx]:.2f}+/-{ukf_std[t_idx]:.2f}")

# Dynamics analysis
print("\n" + "="*60)
print("LEARNED DYNAMICS")
print("="*60)

mean_net.eval()

with torch.no_grad():
    # Test in PHYSICAL space (not normalized) since our dynamics work in physical space
    test_phys = torch.tensor([
        [100.0, 1.0, 50.0, 0.5],    # typical state
        [100.0, 2.0, 50.0, 1.0],    # faster
        [100.0, 0.0, 50.0, 0.0],    # stationary
        [100.0, -1.0, 50.0, -0.5],  # moving backward
    ], device=device, dtype=torch.float32)
    
    pred_phys = mean_net(test_phys)
    
    print(f"\nIn PHYSICAL space (dt={dt}):")
    print("-" * 70)
    
    for i in range(len(test_phys)):
        inp = test_phys[i].cpu().numpy()
        out = pred_phys[i].cpu().numpy()
        delta = out - inp
        
        expected_delta_x = inp[1] * dt  # vx * dt
        expected_delta_y = inp[3] * dt  # vy * dt
        
        print(f"Input: x={inp[0]:.1f}, vx={inp[1]:.2f}, y={inp[2]:.1f}, vy={inp[3]:.2f}")
        print(f"  Learned delta:  dx={delta[0]:.4f}, dvx={delta[1]:.6f}, dy={delta[2]:.4f}, dvy={delta[3]:.6f}")
        print(f"  Expected delta: dx={expected_delta_x:.4f}, dvx=0.0000, dy={expected_delta_y:.4f}, dvy=0.0000")
        print()

print(f"Saved to: {output_dir}")