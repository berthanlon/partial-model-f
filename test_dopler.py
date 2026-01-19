"""
Test script for Neural KF v23 with Doppler measurements
========================================================

FIXED: Baseline UKF loop order corrected (Update -> Predict)
FIXED: Robust Cholesky decomposition added
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import your neural modules
from nn_train import train_v23, NeuralKFv23, SimpleDynamicsNet, FixedCovPredictor, StateScaler, UKFCore, analyze_gradients

torch.set_default_dtype(torch.float32)
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# CONFIG
# =============================================================================
nx = 4  # State: [x, vx, y, vy]
nz = 4  # Measurements: [r1, r2, rdot1, rdot2]
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

# Measurement noise
sigma_r = 1.0
sigma_rdot = 0.5
R_true = np.diag([sigma_r**2, sigma_r**2, sigma_rdot**2, sigma_rdot**2]).astype(np.float32)

x0_mean = np.array([100.0, 1.0, 50.0, 0.5], dtype=np.float32)
P0 = np.diag([10.0, 1.0, 10.0, 1.0]).astype(np.float32)

n_train, n_test, n_timesteps = 2000, 100, 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================
def h_numpy(x):
    px, vx, py, vy = x[0], x[1], x[2], x[3]
    
    # Radar 1
    dx1, dy1 = px - RADAR1[0], py - RADAR1[1]
    r1 = np.sqrt(dx1**2 + dy1**2 + 1e-12)
    dr1 = (dx1 * vx + dy1 * vy) / r1
    
    # Radar 2
    dx2, dy2 = px - RADAR2[0], py - RADAR2[1]
    r2 = np.sqrt(dx2**2 + dy2**2 + 1e-12)
    dr2 = (dx2 * vx + dy2 * vy) / r2
    
    return np.array([r1, r2, dr1, dr2], dtype=np.float32)

def h_torch(x):
    px, vx, py, vy = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    
    dx1, dy1 = px - RADAR1[0], py - RADAR1[1]
    dx2, dy2 = px - RADAR2[0], py - RADAR2[1]
    
    r1 = torch.sqrt(dx1**2 + dy1**2 + 1e-12)
    r2 = torch.sqrt(dx2**2 + dy2**2 + 1e-12)
    
    rdot1 = (dx1 * vx + dy1 * vy) / r1
    rdot2 = (dx2 * vx + dy2 * vy) / r2
    
    return torch.stack([r1, r2, rdot1, rdot2], dim=-1)

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

# =============================================================================
# UKF BASELINE (FIXED)
# =============================================================================
class UKFBaseline:
    def __init__(self, F, Q, R, h_fn, alpha=0.1, beta=2.0, kappa=0.0): # alpha=0.1 for stability
        self.F, self.Q, self.R, self.h = F, Q, R, h_fn
        self.nx = F.shape[0]
        self.nz = R.shape[0]
        
        lmbda = alpha**2 * (self.nx + kappa) - self.nx
        self.gamma = np.sqrt(self.nx + lmbda)
        n = 2 * self.nx + 1
        
        self.Wm = np.zeros(n)
        self.Wc = np.zeros(n)
        self.Wm[0] = lmbda / (self.nx + lmbda)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        self.Wm[1:] = 0.5 / (self.nx + lmbda)
        self.Wc[1:] = 0.5 / (self.nx + lmbda)
        self.n_sigma = n
    
    def _robust_cholesky(self, P):
        """Robust Cholesky with fallback"""
        P = 0.5 * (P + P.T) # Force symmetry
        try:
            return np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-6)
            P_recon = eigvecs @ np.diag(eigvals) @ eigvecs.T
            return np.linalg.cholesky(P_recon)
    
    def run(self, z_seq, x0, P0):
        T = z_seq.shape[0]
        xs = np.zeros((T, self.nx))
        Ps = np.zeros((T, self.nx, self.nx))
        
        x, P = x0.copy(), P0.copy()
        
        for t in range(T):
            # 1. UPDATE FIRST (Correct x_t using z_t)
            L = self._robust_cholesky(P)
            sigma = np.zeros((self.n_sigma, self.nx))
            sigma[0] = x
            for i in range(self.nx):
                sigma[i+1]         = x + self.gamma * L[:, i]
                sigma[self.nx+i+1] = x - self.gamma * L[:, i]
                
            z_sigma = np.array([self.h(s) for s in sigma])
            z_hat = np.sum(self.Wm[:, None] * z_sigma, axis=0)
            
            S = self.R.copy()
            C = np.zeros((self.nx, self.nz))
            for i in range(self.n_sigma):
                dz = z_sigma[i] - z_hat
                dx = sigma[i] - x
                S += self.Wc[i] * np.outer(dz, dz)
                C += self.Wc[i] * np.outer(dx, dz)
            
            # Robust Solve
            try:
                K = C @ np.linalg.inv(S)
            except:
                K = C @ np.linalg.pinv(S)
            
            x = x + K @ (z_seq[t] - z_hat)
            P = P - K @ S @ K.T
            
            xs[t], Ps[t] = x, P
            
            # 2. PREDICT NEXT (Move to t+1)
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
        
        return xs, Ps

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Generating data...")
    gt_train, z_train = generate_data(n_train, n_timesteps)
    gt_test, z_test = generate_data(n_test, n_timesteps)
    
    output_dir = "./test_output_v23_doppler"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING NEURAL KF")
    print("="*60)
    
    # Train Neural KF
    mean_net, cov_net, scaler, history = train_v23(
        z_train=z_train, R=R_true, h_fn=h_torch,
        x0=x0_mean, P0=P0, nx=nx, nz=nz,
        pos_scale=100.0, vel_scale=1.0,
        epochs=400, lr=1e-3, device=str(device),
        checkpoint_path=os.path.join(output_dir, 'neural_kf_v23_doppler.pth'),
        verbose=True
    )
    
    print("\n" + "="*60)
    print("RUNNING BASELINE COMPARISON")
    print("="*60)
    
    # Run Baseline UKF
    ukf_baseline = UKFBaseline(F_true, Q_true, R_true, h_numpy)
    
    ukf_states = []
    for i in range(n_test):
        us, _ = ukf_baseline.run(z_test[i], x0_mean, P0)
        ukf_states.append(us)
    ukf_states = np.array(ukf_states)
    
    # Run Neural KF (using the trained models)
    class NeuralKFv23Fixed:
        def __init__(self, mean_net, cov_net, scaler, R, nx, nz, device):
            self.mean_net = mean_net
            self.cov_net = cov_net
            self.scaler = scaler
            self.ukf = UKFCore(nx, nz, R, device)
            self.device = device
            
        def run(self, z_seq, x0, P0, h_fn):
            self.mean_net.eval(); self.cov_net.eval()
            z_seq = torch.as_tensor(z_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            B, T, _ = z_seq.shape
            x = torch.as_tensor(x0, device=self.device).unsqueeze(0).expand(B, -1)
            P = torch.as_tensor(P0, device=self.device).unsqueeze(0).expand(B, -1, -1)
            states = []
            with torch.no_grad():
                for t in range(T):
                    x_norm = self.scaler.normalize(x)
                    P_norm = self.scaler.scale_cov(P)
                    x_pred = self.scaler.denormalize(self.mean_net(x_norm))
                    P_pred = self.scaler.unscale_cov(self.cov_net(P_norm))
                    x, P, _, _ = self.ukf.update(x_pred, P_pred, z_seq[:, t], h_fn)
                    states.append(x.cpu().numpy())
            return np.array(states).transpose(1, 0, 2)[0]

    neural_kf = NeuralKFv23Fixed(mean_net, cov_net, scaler, R_true, nx, nz, device)
    neural_states = []
    for i in range(n_test):
        ns = neural_kf.run(z_test[i], x0_mean, P0, h_torch)
        neural_states.append(ns)
    neural_states = np.array(neural_states)

    # Compare RMSE
    neural_rmse = np.sqrt(np.mean((neural_states - gt_test)**2))
    ukf_rmse = np.sqrt(np.mean((ukf_states - gt_test)**2))
    
    print(f"\nFinal RMSE Comparison:")
    print(f"Neural KF: {neural_rmse:.4f}")
    print(f"Baseline:  {ukf_rmse:.4f}")