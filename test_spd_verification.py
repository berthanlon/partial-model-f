# test_neural_kf_v7.py
"""
Test v7 - Structured dynamics learning.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from nn_train import train_unsupervised_v7, NeuralKFv7, MeanNetStructured, CovNetSimple

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

n_train = 2000
n_test = 100
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
output_dir = "./test_output_v7"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("TRAINING v7 - STRUCTURED DYNAMICS")
print("="*60)

mean_net, cov_net, history = train_unsupervised_v7(
    z_train=z_train,
    R=R_true,
    h_fn=h_torch,
    x0=x0_mean,
    P0=P0,
    nx=nx,
    nz=nz,
    epochs=1000,
    lr=1e-3,
    device=str(device),
    checkpoint_path=os.path.join(output_dir, 'neural_kf_v7.pth'),
    verbose=True
)

# Plot loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (NLL)')
plt.title(f"Training Loss (best @ epoch {history['best_epoch']})")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['T_curriculum'])
plt.xlabel('Epoch')
plt.ylabel('Sequence Length T')
plt.title('Curriculum')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150)
plt.close()


# =============================================================================
# CORRECT UKF BASELINE
# =============================================================================
class CorrectUKF:
    """Properly implemented UKF baseline."""
    def __init__(self, F, Q, R, h_fn, alpha=0.001, beta=2.0, kappa=0.0):
        self.F, self.Q, self.R, self.h = F, Q, R, h_fn
        self.nx, self.nz = F.shape[0], R.shape[0]
        
        # UKF parameters
        lmbda = alpha**2 * (self.nx + kappa) - self.nx
        self.gamma = np.sqrt(self.nx + lmbda)
        
        n = 2 * self.nx + 1
        self.Wm = np.zeros(n)
        self.Wc = np.zeros(n)
        self.Wm[0] = lmbda / (self.nx + lmbda)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        for i in range(1, n):
            self.Wm[i] = 0.5 / (self.nx + lmbda)
            self.Wc[i] = 0.5 / (self.nx + lmbda)
        self.n_sigma = n
    
    def run(self, z_seq, x0, P0):
        T = z_seq.shape[0]
        xs = np.zeros((T, self.nx))
        Ps = np.zeros((T, self.nx, self.nx))
        
        x = x0.copy()
        P = P0.copy()
        
        for t in range(T):
            # Predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            
            # Sigma points
            P_sqrt = np.linalg.cholesky(P_pred + 1e-6 * np.eye(self.nx))
            sigma = np.zeros((self.n_sigma, self.nx))
            sigma[0] = x_pred
            for i in range(self.nx):
                sigma[i + 1] = x_pred + self.gamma * P_sqrt[:, i]
                sigma[self.nx + i + 1] = x_pred - self.gamma * P_sqrt[:, i]
            
            # Transform through measurement
            z_sigma = np.array([self.h(s) for s in sigma])
            z_hat = np.sum(self.Wm[:, None] * z_sigma, axis=0)
            
            # S and C
            S = self.R.copy()
            C = np.zeros((self.nx, self.nz))
            for i in range(self.n_sigma):
                dz = z_sigma[i] - z_hat
                dx = sigma[i] - x_pred
                S += self.Wc[i] * np.outer(dz, dz)
                C += self.Wc[i] * np.outer(dx, dz)
            
            # Update
            K = C @ np.linalg.inv(S)
            y = z_seq[t] - z_hat
            x = x_pred + K @ y
            P = P_pred - K @ S @ K.T
            
            xs[t] = x
            Ps[t] = P
        
        return xs, Ps


# =============================================================================
# TEST
# =============================================================================
print("\n" + "="*60)
print("TESTING")
print("="*60)

neural_kf = NeuralKFv7(mean_net, cov_net, R_true, device)

test_idx = 0
neural_states, neural_covs = neural_kf.run(z_test[test_idx], x0_mean, P0, h_torch)
neural_states = neural_states[0]

ukf = CorrectUKF(F_true, Q_true, R_true, h_numpy)
ukf_states, ukf_covs = ukf.run(z_test[test_idx], x0_mean, P0)

gt = gt_test[test_idx]

rmse_neural = np.sqrt(np.mean((neural_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))
rmse_ukf = np.sqrt(np.mean((ukf_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))

print(f"\nSingle sequence RMSE:")
print(f"  Neural KF: mean={rmse_neural.mean():.3f}")
print(f"  UKF:       mean={rmse_ukf.mean():.3f}")


# =============================================================================
# PLOTS
# =============================================================================
t_plot = np.arange(n_timesteps)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = ['x (m)', 'vx (m/s)', 'y (m)', 'vy (m/s)']

for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
    ax.plot(t_plot, gt[:, i], 'k-', lw=2, label='Ground Truth')
    ax.plot(t_plot, neural_states[:, i], 'b-', alpha=0.8, label='Neural KF v7')
    ax.plot(t_plot, ukf_states[:, i], 'r--', alpha=0.8, label='UKF')
    ax.set_xlabel('Time Step')
    ax.set_ylabel(lbl)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_comparison.png'), dpi=150)
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(gt[:, 0], gt[:, 2], 'k-', lw=2, label='Ground Truth')
plt.plot(neural_states[:, 0], neural_states[:, 2], 'b-', alpha=0.8, label='Neural KF v7')
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

plt.figure(figsize=(10, 5))
plt.plot(t_plot, rmse_neural, 'b-', lw=2, label='Neural KF v7')
plt.plot(t_plot, rmse_ukf, 'r--', lw=2, label='UKF')
plt.xlabel('Time Step')
plt.ylabel('Position RMSE (m)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'), dpi=150)
plt.close()


# =============================================================================
# MULTI-SEQUENCE
# =============================================================================
print("\n" + "="*60)
print("MULTI-SEQUENCE EVALUATION")
print("="*60)

neural_rmse_all = []
ukf_rmse_all = []

for seq_idx in range(50):
    n_states, _ = neural_kf.run(z_test[seq_idx], x0_mean, P0, h_torch)
    n_states = n_states[0]
    u_states, _ = ukf.run(z_test[seq_idx], x0_mean, P0)
    g = gt_test[seq_idx]
    
    neural_rmse_all.append(np.sqrt(np.mean((n_states[:, [0,2]] - g[:, [0,2]])**2)))
    ukf_rmse_all.append(np.sqrt(np.mean((u_states[:, [0,2]] - g[:, [0,2]])**2)))

print(f"Average RMSE over 50 sequences:")
print(f"  Neural KF v7: {np.mean(neural_rmse_all):.3f} ± {np.std(neural_rmse_all):.3f}")
print(f"  UKF:          {np.mean(ukf_rmse_all):.3f} ± {np.std(ukf_rmse_all):.3f}")

plt.figure(figsize=(8, 6))
plt.boxplot([neural_rmse_all, ukf_rmse_all], labels=['Neural KF v7', 'UKF'])
plt.ylabel('Position RMSE (m)')
plt.title('RMSE Distribution')
plt.grid(True, axis='y')
plt.savefig(os.path.join(output_dir, 'rmse_boxplot.png'), dpi=150)
plt.close()


# =============================================================================
# ANALYZE LEARNED DYNAMICS
# =============================================================================
print("\n" + "="*60)
print("LEARNED DYNAMICS ANALYSIS")
print("="*60)

mean_net.eval()
with torch.no_grad():
    test_states = torch.tensor([
        [100.0, 1.0, 50.0, 0.5],
        [100.0, 2.0, 50.0, 1.0],
        [100.0, 0.0, 50.0, 0.0],
        [100.0, -1.0, 50.0, -0.5],
    ], device=device)
    
    predicted = mean_net(test_states)
    
    print("\nLearned dynamics (should learn dt ≈ 0.5):")
    print("=" * 70)
    
    for i in range(len(test_states)):
        inp = test_states[i].cpu().numpy()
        out = predicted[i].cpu().numpy()
        true_next = F_true @ inp
        
        # Compute implied dt from position change
        if abs(inp[1]) > 0.1:
            implied_dt_x = (out[0] - inp[0]) / inp[1]
        else:
            implied_dt_x = float('nan')
        
        if abs(inp[3]) > 0.1:
            implied_dt_y = (out[2] - inp[2]) / inp[3]
        else:
            implied_dt_y = float('nan')
        
        print(f"\nInput: x={inp[0]:.1f}, vx={inp[1]:.1f}, y={inp[2]:.1f}, vy={inp[3]:.1f}")
        print(f"  Learned next:  x={out[0]:.2f}, vx={out[1]:.2f}, y={out[2]:.2f}, vy={out[3]:.2f}")
        print(f"  True next:     x={true_next[0]:.2f}, vx={true_next[1]:.2f}, y={true_next[2]:.2f}, vy={true_next[3]:.2f}")
        print(f"  Implied dt_x: {implied_dt_x:.3f}, dt_y: {implied_dt_y:.3f} (true: 0.5)")
        print(f"  Vel change: dvx={out[1]-inp[1]:.3f}, dvy={out[3]-inp[3]:.3f} (should be ~0)")

print(f"\nAll saved to: {output_dir}")