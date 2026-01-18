"""
Test script for Neural KF v23 with Doppler measurements
========================================================

This version adds range-rate (Doppler) measurements which directly observe velocity.
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
nx = 4  # State: [x, vx, y, vy]
nz = 4  # Measurements: [r1, r2, rdot1, rdot2] - ranges + Doppler
dt = 0.5
RADAR_BASELINE = 150.0

# Radar positions
RADAR1 = np.array([0.0, 0.0])
RADAR2 = np.array([RADAR_BASELINE, 0.0])

# True dynamics: constant velocity model
F_true = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float32)
sigma_u = 0.01
Q_true = np.array([[dt**3/3, dt**2/2, 0, 0], [dt**2/2, dt, 0, 0],
                   [0, 0, dt**3/3, dt**2/2], [0, 0, dt**2/2, dt]], dtype=np.float32) * sigma_u**2

# Measurement noise: range (1m std), range-rate (0.1 m/s std - Doppler is very accurate)
sigma_r = 1.0
sigma_rdot = 0.5
R_true = np.diag([sigma_r**2, sigma_r**2, sigma_rdot**2, sigma_rdot**2]).astype(np.float32)

x0_mean = np.array([100.0, 1.0, 50.0, 0.5], dtype=np.float32)
P0 = np.diag([10.0, 1.0, 10.0, 1.0]).astype(np.float32)

n_train, n_test, n_timesteps = 2000, 100, 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# MEASUREMENT FUNCTIONS WITH DOPPLER
# =============================================================================

def h_numpy(x):
    """
    Measurement function: range + range-rate to two radars.
    
    State: x = [px, vx, py, vy]
    Output: [r1, r2, rdot1, rdot2]
    
    Range: r = sqrt((px - radar_x)^2 + (py - radar_y)^2)
    Range-rate: rdot = ((px - radar_x)*vx + (py - radar_y)*vy) / r
    """
    px, vx, py, vy = x[0], x[1], x[2], x[3]
    
    # Relative positions to radars
    dx1, dy1 = px - RADAR1[0], py - RADAR1[1]
    dx2, dy2 = px - RADAR2[0], py - RADAR2[1]
    
    # Ranges
    r1 = np.sqrt(dx1**2 + dy1**2 + 1e-12)
    r2 = np.sqrt(dx2**2 + dy2**2 + 1e-12)
    
    # Range-rates (Doppler): rdot = (d/dt)(r) = (rel_pos · velocity) / r
    rdot1 = (dx1 * vx + dy1 * vy) / r1
    rdot2 = (dx2 * vx + dy2 * vy) / r2
    
    return np.array([r1, r2, rdot1, rdot2], dtype=np.float32)

def h_torch(x):
    """Batched measurement function for PyTorch."""
    px, vx, py, vy = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    
    # Relative positions
    dx1, dy1 = px - RADAR1[0], py - RADAR1[1]
    dx2, dy2 = px - RADAR2[0], py - RADAR2[1]
    
    # Ranges
    r1 = torch.sqrt(dx1**2 + dy1**2 + 1e-12)
    r2 = torch.sqrt(dx2**2 + dy2**2 + 1e-12)
    
    # Range-rates
    rdot1 = (dx1 * vx + dy1 * vy) / r1
    rdot2 = (dx2 * vx + dy2 * vy) / r2
    
    return torch.stack([r1, r2, rdot1, rdot2], dim=-1)

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_seq, T):
    """Generate trajectories with Doppler measurements."""
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

# =============================================================================
# UKF BASELINE (Fixed for Doppler)
# =============================================================================

class UKFBaseline:
    """Unscented Kalman Filter with proper Doppler support."""
    
    def __init__(self, F, Q, R, h_fn, alpha=0.001, beta=2.0, kappa=0.0):
        self.F, self.Q, self.R, self.h = F, Q, R, h_fn
        self.nx = F.shape[0]
        self.nz = R.shape[0]
        
        # UKF sigma point parameters
        lmbda = alpha**2 * (self.nx + kappa) - self.nx
        self.gamma = np.sqrt(self.nx + lmbda)
        n = 2 * self.nx + 1
        
        # Weights
        self.Wm = np.zeros(n)
        self.Wc = np.zeros(n)
        self.Wm[0] = lmbda / (self.nx + lmbda)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        self.Wm[1:] = 0.5 / (self.nx + lmbda)
        self.Wc[1:] = 0.5 / (self.nx + lmbda)
        self.n_sigma = n
    
    def _ensure_spd(self, P, min_eig=1e-6):
        """Ensure matrix is symmetric positive definite."""
        P = 0.5 * (P + P.T)
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, min_eig)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def run(self, z_seq, x0, P0):
        """Run UKF on a sequence of measurements."""
        T = z_seq.shape[0]
        xs = np.zeros((T, self.nx))
        Ps = np.zeros((T, self.nx, self.nx))
        x, P = x0.copy(), P0.copy()
        
        for t in range(T):
            # === PREDICTION ===
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            P_pred = self._ensure_spd(P_pred)
            
            # === SIGMA POINTS ===
            try:
                L = np.linalg.cholesky(P_pred)
            except:
                P_pred = self._ensure_spd(P_pred, min_eig=1e-4)
                L = np.linalg.cholesky(P_pred)
            
            sigma = np.zeros((self.n_sigma, self.nx))
            sigma[0] = x_pred
            for i in range(self.nx):
                sigma[i + 1] = x_pred + self.gamma * L[:, i]
                sigma[self.nx + i + 1] = x_pred - self.gamma * L[:, i]
            
            # === MEASUREMENT PREDICTION ===
            z_sigma = np.array([self.h(s) for s in sigma])
            z_hat = np.zeros(self.nz)
            for i in range(self.n_sigma):
                z_hat += self.Wm[i] * z_sigma[i]
            
            # === INNOVATION COVARIANCE ===
            S = self.R.copy()
            C = np.zeros((self.nx, self.nz))
            for i in range(self.n_sigma):
                dz = z_sigma[i] - z_hat
                dx = sigma[i] - x_pred
                S += self.Wc[i] * np.outer(dz, dz)
                C += self.Wc[i] * np.outer(dx, dz)
            
            # Ensure S is SPD
            S = self._ensure_spd(S)
            
            # === UPDATE ===
            try:
                K = C @ np.linalg.inv(S)
            except:
                K = C @ np.linalg.pinv(S)
            
            x = x_pred + K @ (z_seq[t] - z_hat)
            P = P_pred - K @ S @ K.T
            P = self._ensure_spd(P)
            
            xs[t], Ps[t] = x, P
        
        return xs, Ps

# =============================================================================
# GRADIENT ANALYSIS
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
ukf_core = UKFCore(nx, nz, R_true, device)

# Analyze gradients
grads = analyze_gradients(mean_net, cov_net, scaler, ukf_core, z_train_t[:100], x0_mean, P0, h_torch, device)

print("\nGradient magnitudes (larger = more learnable):")
for k, v in grads.items():
    print(f"  {k}: {v:.6f}")

print("\nInterpretation:")
print("  - With Doppler, velocity should have stronger gradient signal")
print("  - d_NLL should be larger than range-only case")

# =============================================================================
# TRAINING
# =============================================================================
output_dir = "./test_output_v23_doppler"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("TRAINING v23 - WITH DOPPLER")
print("="*60)

mean_net, cov_net, scaler, history = train_v23(
    z_train=z_train, R=R_true, h_fn=h_torch,
    x0=x0_mean, P0=P0, nx=nx, nz=nz,
    pos_scale=100.0, vel_scale=1.0,
    epochs=400, lr=1e-3, device=str(device),
    checkpoint_path=os.path.join(output_dir, 'neural_kf_v23_doppler.pth'),
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

# Create a fixed NeuralKFv23 that handles nz=4
# The original class has hardcoded nz=2, so we create the UKFCore separately
class NeuralKFv23Fixed:
    """Fixed version that accepts nz as parameter."""
    def __init__(self, mean_net, cov_net, scaler, R, nx, nz, device):
        self.mean_net = mean_net
        self.cov_net = cov_net
        self.scaler = scaler
        self.ukf = UKFCore(nx, nz, R, device)
        self.device = device
        
    def run(self, z_seq, x0, P0, h_fn):
        self.mean_net.eval()
        self.cov_net.eval()
        
        z_seq = torch.as_tensor(z_seq, dtype=torch.float32, device=self.device)
        if z_seq.dim() == 2:
            z_seq = z_seq.unsqueeze(0)
        B, T, _ = z_seq.shape
        
        x = torch.as_tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1).clone()
        P = torch.as_tensor(P0, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()
        
        states, covs = [], []
        
        with torch.no_grad():
            for t in range(T):
                x_norm = self.scaler.normalize(x)
                P_norm = self.scaler.scale_cov(P)
                
                x_pred_norm = self.mean_net(x_norm)
                P_pred_norm = self.cov_net(P_norm)
                
                x_pred = self.scaler.denormalize(x_pred_norm)
                P_pred = self.scaler.unscale_cov(P_pred_norm)
                
                x, P, _, _ = self.ukf.update(x_pred, P_pred, z_seq[:, t], h_fn)
                
                states.append(x.cpu().numpy())
                covs.append(P.cpu().numpy())
                
        return np.array(states).transpose(1, 0, 2)[0], np.array(covs).transpose(1, 0, 2, 3)[0]

neural_kf = NeuralKFv23Fixed(mean_net, cov_net, scaler, R_true, nx, nz, device)
ukf_baseline = UKFBaseline(F_true, Q_true, R_true, h_numpy)

neural_states, neural_covs = [], []
ukf_states, ukf_covs = [], []

for i in range(n_test):
    ns, nc = neural_kf.run(z_test[i], x0_mean, P0, h_torch)
    neural_states.append(ns)
    neural_covs.append(nc)
    
    us, uc = ukf_baseline.run(z_test[i], x0_mean, P0)
    ukf_states.append(us)
    ukf_covs.append(uc)

neural_states = np.array(neural_states)
neural_covs = np.array(neural_covs)
ukf_states = np.array(ukf_states)
ukf_covs = np.array(ukf_covs)

# Compute RMSE
neural_pos_rmse = np.sqrt(np.mean((neural_states[:, :, [0, 2]] - gt_test[:, :, [0, 2]])**2))
ukf_pos_rmse = np.sqrt(np.mean((ukf_states[:, :, [0, 2]] - gt_test[:, :, [0, 2]])**2))
neural_vel_rmse = np.sqrt(np.mean((neural_states[:, :, [1, 3]] - gt_test[:, :, [1, 3]])**2))
ukf_vel_rmse = np.sqrt(np.mean((ukf_states[:, :, [1, 3]] - gt_test[:, :, [1, 3]])**2))

neural_rmse_all = np.sqrt(np.mean((neural_states[:, :, [0, 2]] - gt_test[:, :, [0, 2]])**2, axis=(1, 2)))
ukf_rmse_all = np.sqrt(np.mean((ukf_states[:, :, [0, 2]] - gt_test[:, :, [0, 2]])**2, axis=(1, 2)))

print(f"Average Position RMSE over {n_test} sequences:")
print(f"  Neural KF v23: {np.mean(neural_rmse_all):.3f} +/- {np.std(neural_rmse_all):.3f}")
print(f"  UKF (oracle):  {np.mean(ukf_rmse_all):.3f} +/- {np.std(ukf_rmse_all):.3f}")

print(f"\nAverage Velocity RMSE:")
print(f"  Neural KF v23: {neural_vel_rmse:.3f}")
print(f"  UKF (oracle):  {ukf_vel_rmse:.3f}")

# NEES
neural_nees, ukf_nees = [], []
for i in range(n_test):
    for t in range(n_timesteps):
        err_n = neural_states[i, t] - gt_test[i, t]
        err_u = ukf_states[i, t] - gt_test[i, t]
        try:
            neural_nees.append(err_n @ np.linalg.inv(neural_covs[i, t] + 1e-6*np.eye(nx)) @ err_n)
            ukf_nees.append(err_u @ np.linalg.inv(ukf_covs[i, t] + 1e-6*np.eye(nx)) @ err_u)
        except:
            pass

print(f"\nNEES (should be ~{nx}):")
print(f"  Neural KF v23: {np.mean(neural_nees):.2f}")
print(f"  UKF (oracle):  {np.mean(ukf_nees):.2f}")

# =============================================================================
# LEARNED DYNAMICS ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("LEARNED DYNAMICS ANALYSIS")
print("="*60)

mean_net.eval()
with torch.no_grad():
    test_cases = [
        ("stationary", [1.0, 0.0, 0.5, 0.0]),      
        ("vx=1,vy=0.5", [1.0, 1.0, 0.5, 0.5]),      
        ("vx=2,vy=1", [1.0, 2.0, 0.5, 1.0]),        
        ("backward", [1.0, -1.0, 0.5, -0.5]),      
    ]
    
    print("\nLearned deltas (normalized space):")
    print("-" * 60)
    
    for name, state in test_cases:
        x_norm = torch.tensor([state], device=device, dtype=torch.float32)
        delta = mean_net.get_delta(x_norm)[0].cpu().numpy()
        
        # Expected delta for CV model (in normalized space)
        # Δx_norm = vx * dt / pos_scale = state[1] * 0.5 / 100 = state[1] * 0.005
        expected_dx = state[1] * dt / 100.0
        
        print(f"  {name:14s}: Δx={delta[0]:.5f} (expect {expected_dx:.5f}), Δvx={delta[1]:.5f}")

# Compute slope
vx_values = np.linspace(-2, 2, 21)
deltas = []
for vx in vx_values:
    x_norm = torch.tensor([[1.0, vx, 0.5, 0.0]], device=device, dtype=torch.float32)
    with torch.no_grad():
        delta = mean_net.get_delta(x_norm)[0, 0].cpu().numpy()
    deltas.append(delta)

deltas = np.array(deltas)
slope, intercept = np.polyfit(vx_values, deltas, 1)
correlation = np.corrcoef(vx_values, deltas)[0, 1]

expected_slope = dt / 100.0  # 0.005

print(f"\nΔx vs vx analysis:")
print(f"  Correlation: {correlation:.3f} (should be ~1.0 for CV)")
print(f"  Slope: {slope:.5f} (should be ~{expected_slope:.5f} for CV)")
print(f"  Intercept: {intercept:.5f} (should be ~0)")

# =============================================================================
# PLOTS
# =============================================================================

t_plot = np.arange(n_timesteps)

# State comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
seq_idx = 0

axes[0, 0].plot(t_plot, gt_test[seq_idx, :, 0], 'k-', lw=2, label='Ground Truth')
axes[0, 0].plot(t_plot, neural_states[seq_idx, :, 0], 'b-', lw=1.5, label='Neural KF v23')
axes[0, 0].plot(t_plot, ukf_states[seq_idx, :, 0], 'r--', lw=1.5, label='UKF (oracle)')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('x (m)')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(t_plot, gt_test[seq_idx, :, 1], 'k-', lw=2, label='Ground Truth')
axes[0, 1].plot(t_plot, neural_states[seq_idx, :, 1], 'b-', lw=1.5, label='Neural KF v23')
axes[0, 1].plot(t_plot, ukf_states[seq_idx, :, 1], 'r--', lw=1.5, label='UKF (oracle)')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('vx (m/s)')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(t_plot, gt_test[seq_idx, :, 2], 'k-', lw=2, label='Ground Truth')
axes[1, 0].plot(t_plot, neural_states[seq_idx, :, 2], 'b-', lw=1.5, label='Neural KF v23')
axes[1, 0].plot(t_plot, ukf_states[seq_idx, :, 2], 'r--', lw=1.5, label='UKF (oracle)')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('y (m)')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(t_plot, gt_test[seq_idx, :, 3], 'k-', lw=2, label='Ground Truth')
axes[1, 1].plot(t_plot, neural_states[seq_idx, :, 3], 'b-', lw=1.5, label='Neural KF v23')
axes[1, 1].plot(t_plot, ukf_states[seq_idx, :, 3], 'r--', lw=1.5, label='UKF (oracle)')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('vy (m/s)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.suptitle('State Estimation (WITH DOPPLER)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'state_comparison.png'), dpi=150)
plt.close()

# Variance comparison
neural_vars_mean = np.mean([np.diagonal(P, axis1=1, axis2=2) for P in neural_covs], axis=0)
ukf_vars_mean = np.mean([np.diagonal(P, axis1=1, axis2=2) for P in ukf_covs], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = ['σ_x', 'σ_vx', 'σ_y', 'σ_vy']
for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
    ax.semilogy(t_plot, neural_vars_mean[:, i], 'b-', lw=2, label='Neural KF v23')
    ax.semilogy(t_plot, ukf_vars_mean[:, i], 'r--', lw=2, label='UKF (oracle)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Variance')
    ax.set_title(f'{lbl} Variance')
    ax.legend()
    ax.grid(True)
plt.suptitle('Variance Comparison (WITH DOPPLER)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'variance_comparison.png'), dpi=150)
plt.close()

# RMSE over time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

neural_rmse_per_seq_t = np.sqrt(np.mean((neural_states[:, :, [0,2]] - gt_test[:, :, [0,2]])**2, axis=2))
ukf_rmse_per_seq_t = np.sqrt(np.mean((ukf_states[:, :, [0,2]] - gt_test[:, :, [0,2]])**2, axis=2))

neural_mean = np.mean(neural_rmse_per_seq_t, axis=0)
neural_std = np.std(neural_rmse_per_seq_t, axis=0)
ukf_mean = np.mean(ukf_rmse_per_seq_t, axis=0)
ukf_std = np.std(ukf_rmse_per_seq_t, axis=0)

axes[0].plot(t_plot, neural_mean, 'b-', lw=2, label='Neural KF v23')
axes[0].fill_between(t_plot, neural_mean - neural_std, neural_mean + neural_std, alpha=0.3, color='blue')
axes[0].plot(t_plot, ukf_mean, 'r--', lw=2, label='UKF (oracle)')
axes[0].fill_between(t_plot, ukf_mean - ukf_std, ukf_mean + ukf_std, alpha=0.3, color='red')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Position RMSE (m)')
axes[0].set_title('Mean Position RMSE ± Std')
axes[0].legend()
axes[0].grid(True)

# Median with percentiles
neural_median = np.median(neural_rmse_per_seq_t, axis=0)
neural_p25 = np.percentile(neural_rmse_per_seq_t, 25, axis=0)
neural_p75 = np.percentile(neural_rmse_per_seq_t, 75, axis=0)
ukf_median = np.median(ukf_rmse_per_seq_t, axis=0)
ukf_p25 = np.percentile(ukf_rmse_per_seq_t, 25, axis=0)
ukf_p75 = np.percentile(ukf_rmse_per_seq_t, 75, axis=0)

axes[1].plot(t_plot, neural_median, 'b-', lw=2, label='Neural KF v23')
axes[1].fill_between(t_plot, neural_p25, neural_p75, alpha=0.3, color='blue')
axes[1].plot(t_plot, ukf_median, 'r--', lw=2, label='UKF (oracle)')
axes[1].fill_between(t_plot, ukf_p25, ukf_p75, alpha=0.3, color='red')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Position RMSE (m)')
axes[1].set_title('Median Position RMSE (25-75 percentile)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rmse_vs_time_all_sequences.png'), dpi=150)
plt.close()

# Calibration check
neural_sq_err = (neural_states - gt_test)**2
ukf_sq_err = (ukf_states - gt_test)**2
neural_mse_per_state = np.mean(neural_sq_err, axis=0)
ukf_mse_per_state = np.mean(ukf_sq_err, axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cal_labels = ['x', 'vx', 'y', 'vy']
for i, (ax, lbl) in enumerate(zip(axes.flat, cal_labels)):
    ax.semilogy(t_plot, neural_mse_per_state[:, i], 'b-', lw=2, label='Neural MSE')
    ax.semilogy(t_plot, neural_vars_mean[:, i], 'b--', lw=2, label='Neural Variance')
    ax.semilogy(t_plot, ukf_mse_per_state[:, i], 'r-', lw=2, label='UKF MSE')
    ax.semilogy(t_plot, ukf_vars_mean[:, i], 'r--', lw=2, label='UKF Variance')
    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'{lbl}²')
    ax.set_title(f'Calibration: {lbl}')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
plt.suptitle('Calibration: MSE vs Predicted Variance (WITH DOPPLER)\n(Well-calibrated: solid ≈ dashed)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'calibration_check.png'), dpi=150)
plt.close()

# NEES over time
neural_nees_per_t = np.zeros((n_test, n_timesteps))
ukf_nees_per_t = np.zeros((n_test, n_timesteps))
for seq_idx in range(n_test):
    for t in range(n_timesteps):
        err_n = neural_states[seq_idx, t] - gt_test[seq_idx, t]
        err_u = ukf_states[seq_idx, t] - gt_test[seq_idx, t]
        P_n = neural_covs[seq_idx, t]
        P_u = ukf_covs[seq_idx, t]
        try:
            neural_nees_per_t[seq_idx, t] = err_n @ np.linalg.inv(P_n + 1e-6*np.eye(nx)) @ err_n
            ukf_nees_per_t[seq_idx, t] = err_u @ np.linalg.inv(P_u + 1e-6*np.eye(nx)) @ err_u
        except:
            neural_nees_per_t[seq_idx, t] = np.nan
            ukf_nees_per_t[seq_idx, t] = np.nan

neural_nees_mean = np.nanmean(neural_nees_per_t, axis=0)
ukf_nees_mean = np.nanmean(ukf_nees_per_t, axis=0)
neural_nees_std = np.nanstd(neural_nees_per_t, axis=0)
ukf_nees_std = np.nanstd(ukf_nees_per_t, axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t_plot, neural_nees_mean, 'b-', lw=2, label='Neural KF v23')
ax.fill_between(t_plot, 
                np.maximum(neural_nees_mean - neural_nees_std, 0),
                neural_nees_mean + neural_nees_std,
                alpha=0.3, color='blue')
ax.plot(t_plot, ukf_nees_mean, 'r--', lw=2, label='UKF (oracle)')
ax.fill_between(t_plot,
                np.maximum(ukf_nees_mean - ukf_nees_std, 0),
                ukf_nees_mean + ukf_nees_std,
                alpha=0.3, color='red')
ax.axhline(nx, color='green', ls=':', lw=2, label=f'Expected (nx={nx})')
ax.axhspan(nx - 2*np.sqrt(2*nx), nx + 2*np.sqrt(2*nx), alpha=0.1, color='green', label='95% interval')
ax.set_xlabel('Time Step')
ax.set_ylabel('NEES')
ax.set_title('NEES Over Time (WITH DOPPLER)')
ax.legend()
ax.grid(True)
ax.set_ylim(0, max(12, np.nanmax(neural_nees_mean + neural_nees_std) * 1.1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'nees_over_time.png'), dpi=150)
plt.close()

# Delta vs velocity plot
plt.figure(figsize=(10, 6))
plt.plot(vx_values, deltas, 'b.-', lw=2, ms=8, label='Learned')
plt.plot(vx_values, expected_slope * vx_values, 'r--', lw=2, label='Expected (CV)')
plt.xlabel('vx (normalized)')
plt.ylabel('Δx (normalized)')
plt.title(f'Does Δx depend on vx?\nCorrelation: {correlation:.3f}, Slope: {slope:.5f}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'delta_vs_velocity.png'), dpi=150)
plt.close()

print(f"\nAll plots saved to: {output_dir}")

# =============================================================================
# DIAGNOSIS
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if correlation > 0.9:
    print("✓ The network learned positive velocity dependence!")
    print(f"  Correlation: {correlation:.3f}, Slope: {slope:.5f} (expected {expected_slope:.5f})")
elif correlation < -0.9:
    print("✗ The network learned NEGATIVE velocity dependence (wrong sign)")
    print(f"  Correlation: {correlation:.3f}, Slope: {slope:.5f} (expected {expected_slope:.5f})")
else:
    print("? The network learned weak/no velocity dependence")
    print(f"  Correlation: {correlation:.3f}")