# test_neural_kf_v8.py
"""
Test v8 - Truncated BPTT for stable training.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from nn_train import train_unsupervised_v8, NeuralKFv8, MeanNetStructured, CovNetSimple

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

sigma_r = 1.0
R_true = np.eye(nz, dtype=np.float32) * sigma_r**2

x0_mean = np.array([100.0, 1.0, 50.0, 0.5], dtype=np.float32)
P0 = np.diag([10.0, 1.0, 10.0, 1.0]).astype(np.float32)

n_train = 2000
n_test = 500
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
# DATA
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
output_dir = "./test_output_v8"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("TRAINING v8 - TRUNCATED BPTT")
print("="*60)

mean_net, cov_net, history = train_unsupervised_v8(
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
    checkpoint_path=os.path.join(output_dir, 'neural_kf_v8.pth'),
    tbptt_len=5,
    verbose=True
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
# UKF BASELINE
# =============================================================================
class CorrectUKF:
    def __init__(self, F, Q, R, h_fn, alpha=0.001, beta=2.0, kappa=0.0):
        self.F, self.Q, self.R, self.h = F, Q, R, h_fn
        self.nx, self.nz = F.shape[0], R.shape[0]
        
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
        x, P = x0.copy(), P0.copy()
        
        for t in range(T):
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            
            P_sqrt = np.linalg.cholesky(P_pred + 1e-6 * np.eye(self.nx))
            sigma = np.zeros((self.n_sigma, self.nx))
            sigma[0] = x_pred
            for i in range(self.nx):
                sigma[i + 1] = x_pred + self.gamma * P_sqrt[:, i]
                sigma[self.nx + i + 1] = x_pred - self.gamma * P_sqrt[:, i]
            
            z_sigma = np.array([self.h(s) for s in sigma])
            z_hat = np.sum(self.Wm[:, None] * z_sigma, axis=0)
            
            S = self.R.copy()
            C = np.zeros((self.nx, self.nz))
            for i in range(self.n_sigma):
                dz = z_sigma[i] - z_hat
                dx = sigma[i] - x_pred
                S += self.Wc[i] * np.outer(dz, dz)
                C += self.Wc[i] * np.outer(dx, dz)
            
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

neural_kf = NeuralKFv8(mean_net, cov_net, R_true, device)

test_idx = 0
neural_states, _ = neural_kf.run(z_test[test_idx], x0_mean, P0, h_torch)
neural_states = neural_states[0]

ukf = CorrectUKF(F_true, Q_true, R_true, h_numpy)
ukf_states, _ = ukf.run(z_test[test_idx], x0_mean, P0)

gt = gt_test[test_idx]

rmse_neural = np.sqrt(np.mean((neural_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))
rmse_ukf = np.sqrt(np.mean((ukf_states[:, [0,2]] - gt[:, [0,2]])**2, axis=1))

print(f"Single sequence RMSE:")
print(f"  Neural KF v8: mean={rmse_neural.mean():.3f}")
print(f"  UKF:          mean={rmse_ukf.mean():.3f}")


# =============================================================================
# PLOTS
# =============================================================================
t_plot = np.arange(n_timesteps)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = ['x (m)', 'vx (m/s)', 'y (m)', 'vy (m/s)']

for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
    ax.plot(t_plot, gt[:, i], 'k-', lw=2, label='Ground Truth')
    ax.plot(t_plot, neural_states[:, i], 'b-', alpha=0.8, label='Neural KF v8')
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
plt.plot(neural_states[:, 0], neural_states[:, 2], 'b-', alpha=0.8, label='Neural KF v8')
plt.plot(ukf_states[:, 0], ukf_states[:, 2], 'r--', alpha=0.8, label='UKF')
plt.plot(0, 0, 'g^', ms=10)
plt.plot(RADAR_BASELINE, 0, 'm^', ms=10)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'xy_trajectory.png'), dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(t_plot, rmse_neural, 'b-', lw=2, label='Neural KF v8')
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
print(f"  Neural KF v8: {np.mean(neural_rmse_all):.3f} ± {np.std(neural_rmse_all):.3f}")
print(f"  UKF:          {np.mean(ukf_rmse_all):.3f} ± {np.std(ukf_rmse_all):.3f}")

plt.figure(figsize=(8, 6))
plt.boxplot([neural_rmse_all, ukf_rmse_all], labels=['Neural KF v8', 'UKF'])
plt.ylabel('Position RMSE (m)')
plt.grid(True, axis='y')
plt.savefig(os.path.join(output_dir, 'rmse_boxplot.png'), dpi=150)
plt.close()


# =============================================================================
# DYNAMICS ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("LEARNED DYNAMICS")
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
    
    print("\nLearned vs True dynamics (dt=0.5):")
    print("-" * 70)
    
    for i in range(len(test_states)):
        inp = test_states[i].cpu().numpy()
        out = predicted[i].cpu().numpy()
        true_next = F_true @ inp
        
        if abs(inp[1]) > 0.1:
            implied_dt_x = (out[0] - inp[0]) / inp[1]
        else:
            implied_dt_x = float('nan')
        
        if abs(inp[3]) > 0.1:
            implied_dt_y = (out[2] - inp[2]) / inp[3]
        else:
            implied_dt_y = float('nan')
        
        print(f"vx={inp[1]:+.1f}, vy={inp[3]:+.1f} | "
              f"dt_x={implied_dt_x:.3f}, dt_y={implied_dt_y:.3f} | "
              f"dvx={out[1]-inp[1]:+.4f}, dvy={out[3]-inp[3]:+.4f}")

print(f"\nSaved to: {output_dir}")

def plot_velocity_diagnosis(neural_kf, z_test, x0, P0, h_fn, index=0):
    """
    Diagnoses if the velocity jitter is due to the Network or the Filter.
    """
    import matplotlib.pyplot as plt
    
    # 1. Run the Full Filter (Neural KF)
    states_filter, _ = neural_kf.run(z_test[index:index+1], x0[index], P0, h_fn)
    states_filter = states_filter[0] # Shape (T, 4)
    
    # 2. Run Open-Loop Prediction (What the Net 'wants' to do without correction)
    # We manually step the mean_net forward without the UKF update
    mean_net = neural_kf.mean_net
    scaler = neural_kf.mean_net.scaler # Access the scaler inside the wrapper
    
    # Prepare initial state
    current_x_phys = torch.tensor(x0[index], device=neural_kf.device).unsqueeze(0)
    pred_states = [current_x_phys.cpu().numpy()[0]]
    
    with torch.no_grad():
        for t in range(len(z_test[index])):
            # Normalize
            x_norm = scaler.normalize(current_x_phys)
            # Predict
            x_new_norm = mean_net.net(x_norm) # Access inner net
            # Denormalize
            current_x_phys = scaler.denormalize(x_new_norm)
            pred_states.append(current_x_phys.cpu().numpy()[0])
            
    pred_states = np.array(pred_states)
    
    # 3. Plotting
    time = np.arange(len(states_filter)) * 0.5
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Position X
    ax[0].plot(time, states_filter[:, 0], 'b-', label='Filter Output (UKF)', linewidth=2)
    ax[0].plot(time, pred_states[1:, 0], 'r--', label='Pure Net Prediction (Open Loop)', linewidth=2)
    ax[0].set_title(f"Position X (Trajectory {index})")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (m)")
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot Velocity X
    ax[1].plot(time, states_filter[:, 1], 'b-', label='Filter Velocity', linewidth=2)
    ax[1].plot(time, pred_states[1:, 1], 'r--', label='Pure Net Velocity', linewidth=2)
    ax[1].set_title(f"Velocity X (Trajectory {index})")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].legend()
    ax[1].grid(True)
    
    plt.suptitle("Diagnosis: Is the Network Smooth?", fontsize=16)
    plt.tight_layout()
    plt.show()

# Example Usage (assuming you have loaded 'neural_kf' and data)
plot_velocity_diagnosis(neural_kf, z_test, x0_mean, P0, h_torch, index=5)