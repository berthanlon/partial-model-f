# test_neural_kf_v5.py
"""
Test the v5 Neural KF that trains WITH the filter in the loop.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from nn_train import train_neural_kf, NeuralKF, MeanNet, CovNet, UKFCore

torch.set_default_dtype(torch.float32)
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# CONFIG
# =============================================================================
nx = 4
nz = 2
T_sample = 0.5
RADAR_BASELINE = 150.0

F_true = np.array([
    [1, T_sample, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, T_sample],
    [0, 0, 0, 1]
], dtype=np.float32)

sigma_u = 0.01
Q_true = np.array([
    [T_sample**3/3, T_sample**2/2, 0, 0],
    [T_sample**2/2, T_sample, 0, 0],
    [0, 0, T_sample**3/3, T_sample**2/2],
    [0, 0, T_sample**2/2, T_sample]
], dtype=np.float32) * sigma_u**2

sigma_r = 2.0
R_true = np.eye(nz, dtype=np.float32) * sigma_r**2

x0_mean = np.array([100.0, 1.0, 50.0, 0.5], dtype=np.float32)
P0 = np.diag([10.0, 1.0, 10.0, 1.0]).astype(np.float32)

n_train = 999
n_test = 200
n_timesteps = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =============================================================================
# MEASUREMENT MODEL
# =============================================================================
def h_numpy(x):
    r1 = np.sqrt(x[0]**2 + x[2]**2 + 1e-12)
    r2 = np.sqrt((x[0] - RADAR_BASELINE)**2 + x[2]**2 + 1e-12)
    return np.array([r1, r2], dtype=np.float32)

def h_torch(x):
    r1 = torch.sqrt(x[:, 0]**2 + x[:, 2]**2 + 1e-12)
    r2 = torch.sqrt((x[:, 0] - RADAR_BASELINE)**2 + x[:, 2]**2 + 1e-12)
    return torch.stack([r1, r2], dim=-1)


# =============================================================================
# DATA GENERATION
# =============================================================================
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


# =============================================================================
# TRAIN
# =============================================================================
output_dir = "./test_output_v5"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("TRAINING NEURAL KF (v5 - with filter in loop)")
print("="*60)

mean_net, cov_net, history = train_neural_kf(
    z_train=z_train,
    F=F_true,
    Q=Q_true,
    R=R_true,
    h_fn=h_torch,
    x0=x0_mean,
    P0=P0,
    epochs=200,
    lr=1e-3,
    device=str(device),
    checkpoint_path=os.path.join(output_dir, 'neural_kf_v5.pth')
)

# Plot loss
plt.figure(figsize=(10, 4))
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (NLL)')
plt.title(f"Training Loss (best @ epoch {history['best_epoch']})")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150)
plt.close()


# =============================================================================
# SIMPLE UKF BASELINE
# =============================================================================
class SimpleUKF:
    def __init__(self, F, Q, R, h_fn):
        self.F, self.Q, self.R, self.h = F, Q, R, h_fn
        self.nx, self.nz = F.shape[0], R.shape[0]
        
        alpha, beta, kappa = 0.001, 2.0, 0.0
        lmbda = alpha**2 * (self.nx + kappa) - self.nx
        self.gamma = np.sqrt(self.nx + lmbda)
        
        n = 2 * self.nx + 1
        self.Wm = np.full(n, 0.5 / (self.nx + lmbda))
        self.Wm[0] = lmbda / (self.nx + lmbda)
        self.Wc = self.Wm.copy()
        self.Wc[0] += (1 - alpha**2 + beta)
        self.n_sigma = n
    
    def run(self, z_seq, x0, P0):
        T = z_seq.shape[0]
        xs, Ps = np.zeros((T, self.nx)), np.zeros((T, self.nx, self.nx))
        x, P = x0.copy(), P0.copy()
        
        for t in range(T):
            # Predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            
            # Sigma points
            try:
                L = np.linalg.cholesky((self.nx + self.gamma**2/(2*self.nx+1)) * P_pred + 1e-6*np.eye(self.nx))
            except:
                L = np.eye(self.nx)
            
            sigma = np.zeros((self.n_sigma, self.nx))
            sigma[0] = x_pred
            for i in range(self.nx):
                sigma[i+1] = x_pred + self.gamma * L[:, i]
                sigma[self.nx+i+1] = x_pred - self.gamma * L[:, i]
            
            # Transform
            z_sigma = np.array([self.h(s) for s in sigma])
            z_hat = self.Wm @ z_sigma
            
            # S, C
            S = self.R.copy()
            C = np.zeros((self.nx, self.nz))
            for i in range(self.n_sigma):
                dz = z_sigma[i] - z_hat
                dx = sigma[i] - x_pred
                S += self.Wc[i] * np.outer(dz, dz)
                C += self.Wc[i] * np.outer(dx, dz)
            
            # Update
            K = C @ np.linalg.inv(S)
            x = x_pred + K @ (z_seq[t] - z_hat)
            P = P_pred - K @ S @ K.T
            
            xs[t], Ps[t] = x, P
        
        return xs, Ps


# =============================================================================
# TEST
# =============================================================================
print("\n" + "="*60)
print("TESTING")
print("="*60)

# Neural KF
neural_kf = NeuralKF(mean_net, cov_net, R_true, device)

test_idx = 0
neural_states, neural_covs = neural_kf.run(z_test[test_idx], x0_mean, P0, h_torch)
neural_states = neural_states[0]  # Remove batch dim
neural_covs = neural_covs[0]

# UKF baseline
ukf = SimpleUKF(F_true, Q_true, R_true, h_numpy)
ukf_states, ukf_covs = ukf.run(z_test[test_idx], x0_mean, P0)

# Ground truth
gt = gt_test[test_idx]

# RMSE
rmse_neural = np.sqrt(np.mean((neural_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))
rmse_ukf = np.sqrt(np.mean((ukf_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))

print(f"RMSE (position):")
print(f"  Neural KF: mean={rmse_neural.mean():.3f}, final={rmse_neural[-1]:.3f}")
print(f"  UKF:       mean={rmse_ukf.mean():.3f}, final={rmse_ukf[-1]:.3f}")


# =============================================================================
# PLOTS
# =============================================================================
t_plot = np.arange(n_timesteps)

# State comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = ['x (m)', 'vx (m/s)', 'y (m)', 'vy (m/s)']

for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
    ax.plot(t_plot, gt[:, i], 'k-', lw=2, label='Ground Truth')
    ax.plot(t_plot, neural_states[:, i], 'b-', alpha=0.8, label='Neural KF')
    ax.plot(t_plot, ukf_states[:, i], 'r--', alpha=0.8, label='UKF')
    ax.set_xlabel('Time Step')
    ax.set_ylabel(lbl)
    ax.legend()
    ax.grid(True)
    ax.set_title(f'State: {lbl}')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_comparison.png'), dpi=150)
plt.close()

# XY trajectory
plt.figure(figsize=(8, 8))
plt.plot(gt[:, 0], gt[:, 2], 'k-', lw=2, label='Ground Truth')
plt.plot(neural_states[:, 0], neural_states[:, 2], 'b-', alpha=0.8, label='Neural KF')
plt.plot(ukf_states[:, 0], ukf_states[:, 2], 'r--', alpha=0.8, label='UKF')
plt.plot(0, 0, 'g^', ms=10, label='Radar 1')
plt.plot(RADAR_BASELINE, 0, 'm^', ms=10, label='Radar 2')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid(True)
plt.title('XY Trajectory')
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'xy_trajectory.png'), dpi=150)
plt.close()

# RMSE
plt.figure(figsize=(10, 5))
plt.plot(t_plot, rmse_neural, 'b-', lw=2, label='Neural KF')
plt.plot(t_plot, rmse_ukf, 'r--', lw=2, label='UKF')
plt.xlabel('Time Step')
plt.ylabel('Position RMSE (m)')
plt.legend()
plt.grid(True)
plt.title('Position RMSE vs Time')
plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'), dpi=150)
plt.close()

# Test on multiple sequences
print("\n" + "="*60)
print("MULTI-SEQUENCE EVALUATION")
print("="*60)

neural_rmse_all = []
ukf_rmse_all = []

for seq_idx in range(min(20, n_test)):
    n_states, _ = neural_kf.run(z_test[seq_idx], x0_mean, P0, h_torch)
    n_states = n_states[0]
    u_states, _ = ukf.run(z_test[seq_idx], x0_mean, P0)
    g = gt_test[seq_idx]
    
    neural_rmse_all.append(np.sqrt(np.mean((n_states[:, [0,2]] - g[:, [0,2]])**2)))
    ukf_rmse_all.append(np.sqrt(np.mean((u_states[:, [0,2]] - g[:, [0,2]])**2)))

print(f"Average RMSE over {len(neural_rmse_all)} sequences:")
print(f"  Neural KF: {np.mean(neural_rmse_all):.3f} ± {np.std(neural_rmse_all):.3f}")
print(f"  UKF:       {np.mean(ukf_rmse_all):.3f} ± {np.std(ukf_rmse_all):.3f}")

print(f"\nAll saved to: {output_dir}")