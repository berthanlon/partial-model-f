# -*- coding: utf-8 -*-
"""
Parameters 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

RADAR_BASELINE: float = 150.0  # second radar at (150, 0)
_EPS: float = 1e-12
#######################
### Size of DataSet ###
#######################

# Number of Training Examples
N_E = 5000

# Number of Cross Validation Examples
N_CV = 20

N_T = 100

# Sequence Length for Linear Case
n_steps = 20

#########################################
##### Neural network parameters #########
#########################################

# mean_ini = np.array([100, 1, 0, 2], dtype = np.float32)
# P_ini   = np.diag([1, 0.1, 1, 0.1]).astype(np.float32)

mean_ini = np.array([100, 1, 0, 0.5], dtype=np.float32)  # start well above y=0

P_ini = np.diag([1, 0.1, 0.5, 0.1]).astype(np.float32)     # a bit less spread in y


chol_ini = np.linalg.cholesky(P_ini)

# Nearly constant velocity model
T = 0.5 # sampling time
F = np.array([[1, T, 0, 0], 
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]], dtype = np.float32) # state transition matrix

sigma_u = 0.001 # standard deviation of the acceleration    ####
Q = np.array([[T**3/3, T**2/2, 0, 0], 
              [T**2/2, T, 0, 0],
              [0, 0, T**3/3, T**2/2],
              [0, 0, T**2/2, T]], dtype = np.float32) * sigma_u**2 # covariance of the process noise
chol_Q = np.linalg.cholesky(Q) # Cholesky decomposition of Q
Q_inv = np.linalg.inv(Q)
chol_Q_inv = np.linalg.cholesky(Q) #_inv)
A = torch.from_numpy(chol_Q_inv)

sigma_r = 0.1
R = np.diag([2,2]) # covariance of the measurement noise
R_inv = np.linalg.inv(R)
chol_R = np.linalg.cholesky(R) #_inv) # Cholesky decomposition of R

B = chol_R # torch.from_numpy(chol_R).float().to(self.device)

m = 4
n = 2

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_batched_time_major(z) -> np.ndarray:
    """Ensure measurement arrays are (B,T,nz) from (B,nz,T) or (T,nz)."""
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    z = np.asarray(z)
    if z.ndim == 3:
        return np.transpose(z, (0, 2, 1)).astype(np.float32, copy=False)
    if z.ndim == 2:
        return z[None, ...].astype(np.float32, copy=False)
    raise ValueError(f"Unexpected measurement shape {z.shape}")

# ---- Torch measurement & Jacobian ----
def h_torch(x: torch.Tensor) -> torch.Tensor:
    # x: (B,nx) with [x, xdot, y, ydot]
    x_pos = x[:, 0]; y_pos = x[:, 2]
    r1 = torch.sqrt(x_pos**2 + y_pos**2 + _EPS)
    r2 = torch.sqrt((x_pos - RADAR_BASELINE)**2 + y_pos**2 + _EPS)
    return torch.stack([r1, r2], dim=-1)

def H_jac_torch(x: torch.Tensor) -> torch.Tensor:
    B_, nx_ = x.shape[0], x.shape[1]
    x_pos = x[:, 0]; y_pos = x[:, 2]
    r1 = torch.sqrt(x_pos**2 + y_pos**2 + _EPS)
    r2 = torch.sqrt((x_pos - RADAR_BASELINE)**2 + y_pos**2 + _EPS)
    H = torch.zeros(B_, 2, nx_, device=x.device, dtype=x.dtype)
    H[:, 0, 0] = x_pos / r1;                H[:, 0, 2] = y_pos / r1
    H[:, 1, 0] = (x_pos - RADAR_BASELINE) / r2; H[:, 1, 2] = y_pos / r2
    return H

# ---- NumPy measurement for simulator ----
def h(x: np.ndarray) -> np.ndarray:
    x_pos = x[0]; y_pos = x[2]
    r1 = np.sqrt(x_pos**2 + y_pos**2)
    r2 = np.sqrt((x_pos - RADAR_BASELINE)**2 + y_pos**2)
    return np.array([r1, r2], dtype=np.float32)

def make_R_full(R_in, nz: int) -> np.ndarray:
    R_np = np.array(R_in, dtype=np.float32)
    if R_np.ndim == 0:
        R_full = np.diag(np.full(nz, R_np, dtype=np.float32))
    elif R_np.ndim == 1:
        R_full = np.diag(R_np.astype(np.float32))
    else:
        R_full = R_np
    return R_full + 1e-8 * np.eye(nz, dtype=np.float32)

def reshape_data(data):
    # If the input is a PyTorch tensor, move to CPU if necessary and convert to NumPy
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Perform the transpose operation
    data_transposed = data.transpose(0, 2, 1)  # Swap the last two dimensions
    
    # Convert back to a PyTorch tensor
    return torch.tensor(data_transposed, dtype=torch.float32)