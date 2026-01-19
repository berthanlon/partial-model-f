# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 01:30:06 2026

@author: betti
"""

"""
test_baseline.py
================
Isolated test bench for the Analytical UKF (Baseline).
Fixes the 'Matrix not positive definite' crash by:
1. Correcting the loop order (Update -> Predict) to match data generation.
2. Using robust Cholesky decomposition with fallback.
3. Enforcing symmetry and positive eigenvalues explicitly.
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG & PHYSICS
# =============================================================================
np.random.seed(42)

nx = 4  # [x, vx, y, vy]
nz = 4  # [r1, r2, dr1, dr2]
dt = 0.5
RADAR_BASELINE = 150.0

# 1. Define True Physics (CV Model)
F_true = np.array([
    [1, dt, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1]
], dtype=np.float64)

# Process Noise (Continuous White Noise approx)
sigma_u = 0.01
Q_true = np.array([
    [dt**3/3, dt**2/2, 0, 0],
    [dt**2/2, dt, 0, 0],
    [0, 0, dt**3/3, dt**2/2],
    [0, 0, dt**2/2, dt]
], dtype=np.float64) * sigma_u**2

# Measurement Noise
sigma_r = 1.0      # Range error (m)
sigma_rdot = 0.1   # Doppler error (m/s)
R_true = np.diag([sigma_r**2, sigma_r**2, sigma_rdot**2, sigma_rdot**2]).astype(np.float64)

# Initialization
x0_mean = np.array([100.0, 1.0, 50.0, 0.5])
P0 = np.diag([10.0, 1.0, 10.0, 1.0])

# =============================================================================
# MEASUREMENT FUNCTION
# =============================================================================
def h_numpy(x):
    px, vx, py, vy = x[0], x[1], x[2], x[3]
    
    # Radar 1 (Origin)
    dx1, dy1 = px - 0.0, py - 0.0
    r1 = np.sqrt(dx1**2 + dy1**2 + 1e-12)
    dr1 = (dx1 * vx + dy1 * vy) / r1
    
    # Radar 2 (Baseline)
    dx2, dy2 = px - RADAR_BASELINE, py - 0.0
    r2 = np.sqrt(dx2**2 + dy2**2 + 1e-12)
    dr2 = (dx2 * vx + dy2 * vy) / r2
    
    return np.array([r1, r2, dr1, dr2])

# =============================================================================
# ROBUST UKF CLASS
# =============================================================================
class RobustUKF:
    def __init__(self, F, Q, R, h_fn, alpha=0.1, beta=2.0, kappa=0.0):
        self.F, self.Q, self.R, self.h = F, Q, R, h_fn
        self.nx = F.shape[0]
        self.nz = R.shape[0]
        
        # Sigma Point Parameters
        # alpha=0.1 is safer for float32/64 than 0.001 to prevent numerical collapse
        self.alpha = alpha 
        self.beta = beta
        self.kappa = kappa
        self.lmbda = self.alpha**2 * (self.nx + self.kappa) - self.nx
        self.gamma = np.sqrt(self.nx + self.lmbda)
        
        # Weights
        self.n_sigma = 2 * self.nx + 1
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        self.Wm[0] = self.lmbda / (self.nx + self.lmbda)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wm[1:] = 0.5 / (self.nx + self.lmbda)
        self.Wc[1:] = 0.5 / (self.nx + self.lmbda)

    def _robust_cholesky(self, P):
        """
        Attempts Cholesky. If fails, forces positive definiteness.
        """
        # 1. Enforce Symmetry
        P = 0.5 * (P + P.T)
        
        try:
            return np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # 2. Eigendecomposition Fallback
            eigvals, eigvecs = np.linalg.eigh(P)
            # Clip negative eigenvalues to small positive number
            eigvals = np.maximum(eigvals, 1e-6)
            # Reconstruct
            P_recon = eigvecs @ np.diag(eigvals) @ eigvecs.T
            return np.linalg.cholesky(P_recon)

    def run(self, z_seq, x0, P0):
        T = len(z_seq)
        xs = np.zeros((T, self.nx))
        Ps = np.zeros((T, self.nx, self.nx))
        
        x_curr = x0.copy()
        P_curr = P0.copy()
        
        for t in range(T):
            # =========================================================
            # STEP 1: UPDATE (Correct x_t using z_t)
            # =========================================================
            
            # A. Generate Sigma Points from Current Estimate
            L = self._robust_cholesky(P_curr)
            sigma = np.zeros((self.n_sigma, self.nx))
            sigma[0] = x_curr
            for i in range(self.nx):
                sigma[i+1]         = x_curr + self.gamma * L[:, i]
                sigma[self.nx+i+1] = x_curr - self.gamma * L[:, i]
                
            # B. Transform Sigma Points through Measurement Function
            z_sigma = np.array([self.h(s) for s in sigma])
            
            # C. Measurement Mean
            z_hat = np.sum(self.Wm[:, None] * z_sigma, axis=0)
            
            # D. Innovation Covariance (S) & Cross Covariance (C)
            S = self.R.copy()
            C = np.zeros((self.nx, self.nz))
            
            for i in range(self.n_sigma):
                dz = z_sigma[i] - z_hat
                dx = sigma[i] - x_curr
                S += self.Wc[i] * np.outer(dz, dz)
                C += self.Wc[i] * np.outer(dx, dz)
                
            # E. Kalman Gain
            # Use pseudo-inverse for stability
            K = C @ np.linalg.pinv(S)
            
            # F. Update State and Covariance
            x_upd = x_curr + K @ (z_seq[t] - z_hat)
            P_upd = P_curr - K @ S @ K.T
            
            # Store results for this timestep
            xs[t] = x_upd
            Ps[t] = P_upd
            
            # =========================================================
            # STEP 2: PREDICT (Move to t+1)
            # =========================================================
            # Standard Linear Prediction (since F is linear here)
            # Note: If F was non-linear, we would do sigma points again here.
            x_curr = self.F @ x_upd
            P_curr = self.F @ P_upd @ self.F.T + self.Q
            
        return xs, Ps

# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_data(n_timesteps=100):
    states = np.zeros((n_timesteps, nx))
    measurements = np.zeros((n_timesteps, nz))
    
    # Random initial state
    x = x0_mean + np.random.randn(nx) * [2.0, 0.5, 2.0, 0.5]
    
    for t in range(n_timesteps):
        # 1. Record State
        states[t] = x
        
        # 2. Generate Measurement (at current state)
        z = h_numpy(x) + np.random.randn(nz) @ np.linalg.cholesky(R_true).T
        measurements[t] = z
        
        # 3. Evolve State (to next timestep)
        w = np.random.randn(nx) @ np.linalg.cholesky(Q_true).T
        x = F_true @ x + w
        
    return states, measurements

# =============================================================================
# MAIN TEST
# =============================================================================
if __name__ == "__main__":
    print("Generating Data...")
    gt, obs = generate_data(n_timesteps=100)
    
    print("Running Robust Baseline UKF...")
    ukf = RobustUKF(F_true, Q_true, R_true, h_numpy)
    
    try:
        est, covs = ukf.run(obs, x0_mean, P0)
        print("Success! UKF ran without crashing.")
    except Exception as e:
        print(f"FAILED: {e}")
        exit()

    # --- METRICS ---
    pos_err = np.sqrt(np.mean((est[:, [0,2]] - gt[:, [0,2]])**2, axis=1))
    vel_err = np.sqrt(np.mean((est[:, [1,3]] - gt[:, [1,3]])**2, axis=1))
    
    mean_pos_rmse = np.mean(pos_err)
    mean_vel_rmse = np.mean(vel_err)
    
    print(f"\nResults:")
    print(f"Position RMSE: {mean_pos_rmse:.4f} meters")
    print(f"Velocity RMSE: {mean_vel_rmse:.4f} m/s")
    
    if mean_pos_rmse < 1.0:
        print(">> VERDICT: Baseline is working correctly.")
    else:
        print(">> VERDICT: Baseline is inaccurate (High RMSE).")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(gt[:, 0], 'k--', label='True X')
    plt.plot(est[:, 0], 'b', label='Est X')
    plt.title('X Position')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(gt[:, 1], 'k--', label='True Vx')
    plt.plot(est[:, 1], 'b', label='Est Vx')
    plt.title('X Velocity')
    
    plt.subplot(2, 2, 3)
    plt.plot(gt[:, 2], 'k--', label='True Y')
    plt.plot(est[:, 2], 'b', label='Est Y')
    plt.title('Y Position')
    
    plt.subplot(2, 2, 4)
    plt.plot(gt[:, 3], 'k--', label='True Vy')
    plt.plot(est[:, 3], 'b', label='Est Vy')
    plt.title('Y Velocity')
    
    plt.tight_layout()
    plt.show()