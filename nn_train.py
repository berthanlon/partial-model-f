# nn_train.py
"""
FULLY UNSUPERVISED Neural Kalman Filter - v8 + Data Scaling
Fixes:
1. Returns (mean_net, cov_net, history) to match test script.
2. Auto-wraps networks to handle scaling transparently.
3. FIX: Checks if loss.requires_grad before backward() to prevent crashes on fail-safe.
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func

# =============================================================================
# DATA SCALER UTILITY
# =============================================================================
class DataScaler:
    def __init__(self, z_data, device='cpu'):
        # z_data shape: (B, T, nz)
        max_val = torch.max(z_data) if z_data.numel() > 0 else torch.tensor(100.0)
        
        # Heuristic Mean: Center positions, assume zero velocity on average
        self.mean = torch.tensor([max_val/2, 0.0, max_val/2, 0.0], device=device)
        self.std = torch.tensor([max_val/2, 50.0, max_val/2, 50.0], device=device)
        
        self.std_mat = torch.diag(self.std)
        self.inv_std_mat = torch.diag(1.0 / self.std)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x_norm):
        return x_norm * self.std + self.mean

    def scale_P_down(self, P_phys):
        return self.inv_std_mat @ P_phys @ self.inv_std_mat.T

    def scale_P_up(self, P_norm):
        return self.std_mat @ P_norm @ self.std_mat.T

# =============================================================================
# WRAPPERS (Make Scaled Nets look like Standard Nets)
# =============================================================================
class ScaledMeanNetWrapper(nn.Module):
    def __init__(self, net, scaler):
        super().__init__()
        self.net = net
        self.scaler = scaler
        self.nx = net.nx

    def forward(self, x_phys):
        x_norm = self.scaler.normalize(x_phys)
        x_new_norm = self.net(x_norm)
        return self.scaler.denormalize(x_new_norm)

class ScaledCovNetWrapper(nn.Module):
    def __init__(self, net, scaler):
        super().__init__()
        self.net = net
        self.scaler = scaler
        self.nx = net.nx

    def forward(self, P_phys):
        P_norm = self.scaler.scale_P_down(P_phys)
        P_new_norm = self.net(P_norm)
        return self.scaler.scale_P_up(P_new_norm)

# =============================================================================
# NETWORKS
# =============================================================================

class MeanNetStructured(nn.Module):
    def __init__(self, nx, hidden=64):
        super().__init__()
        self.nx = nx
        self.vel_to_pos = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.vel_update = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.vel_to_pos:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3); nn.init.zeros_(m.bias)
        for m in self.vel_update:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    
    def forward(self, x):
        pos_x, vel_x = x[:, 0:1], x[:, 1:2]
        pos_y, vel_y = x[:, 2:3], x[:, 3:4]
        dx = self.vel_to_pos(vel_x)
        dy = self.vel_to_pos(vel_y)
        dvx = torch.clamp(self.vel_update(vel_x), -0.1, 0.1)
        dvy = torch.clamp(self.vel_update(vel_y), -0.1, 0.1)
        return torch.cat([pos_x+dx, vel_x+dvx, pos_y+dy, vel_y+dvy], dim=1)

class CovNetSimple(nn.Module):
    def __init__(self, nx, init_std=0.000001): # Changed default init to be smaller
        super().__init__()
        self.nx = nx
        # Initialize small to encourage smoothness early on
        self.log_L_diag = nn.Parameter(torch.full((nx,), np.log(init_std)))
        
        n_lower = nx * (nx - 1) // 2
        self.L_lower = nn.Parameter(torch.zeros(n_lower))
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
    
    def forward(self, P_prev):
        B = P_prev.shape[0]
        device = P_prev.device
        
        # FIX: Lower the minimum clamp significantly
        # 0.1 was too high for normalized space (caused 2500m^2 variance)
        # 1e-4 allows the network to learn "tight" physics
        L_diag = torch.exp(self.log_L_diag)
        L_diag = torch.clamp(L_diag, min=1e-4, max=10.0) 
        
        L = torch.zeros(B, self.nx, self.nx, device=device)
        L[:, range(self.nx), range(self.nx)] = L_diag.unsqueeze(0).expand(B, -1)
        
        if self.L_lower.numel() > 0:
            scale = torch.sqrt(L_diag[self.tril_idx[0]] * L_diag[self.tril_idx[1]])
            L_low = torch.tanh(self.L_lower) * 0.5 * scale
            L[:, self.tril_idx[0], self.tril_idx[1]] = L_low.unsqueeze(0).expand(B, -1)
        
        P = L @ L.transpose(-1, -2)
        # Add a tiny jitter for numerical safety, but much smaller than before
        return P + 1e-6 * torch.eye(self.nx, device=device).unsqueeze(0)
    
# =============================================================================
# UKF CORE
# =============================================================================

class UKFCore:
    def __init__(self, nx, nz, R, device, alpha=0.001, beta=2.0, kappa=0.0):
        self.nx, self.nz, self.device = nx, nz, device
        self.R = torch.as_tensor(R, dtype=torch.float32, device=device)
        if self.R.dim() == 2: self.R = self.R.unsqueeze(0)
        lmbda = alpha**2 * (nx + kappa) - nx
        self.gamma = np.sqrt(nx + lmbda)
        self.n_sigma = 2 * nx + 1
        Wm = torch.zeros(self.n_sigma, device=device)
        Wc = torch.zeros(self.n_sigma, device=device)
        Wm[0] = lmbda / (nx + lmbda)
        Wc[0] = Wm[0] + (1 - alpha**2 + beta)
        Wm[1:] = 0.5 / (nx + lmbda); Wc[1:] = Wm[1:]
        self.Wm, self.Wc = Wm, Wc
    
    def _safe_cholesky(self, P):
        try: return torch.linalg.cholesky(P + 1e-4 * torch.eye(self.nx, device=self.device))
        except:
            e, v = torch.linalg.eigh(P + 1e-4 * torch.eye(self.nx, device=self.device))
            return v @ torch.diag_embed(torch.sqrt(torch.clamp(e, min=1e-4)))
    
    def update(self, x_pred, P_pred, z, h_fn):
        B = x_pred.shape[0]
        sigma = torch.zeros(B, self.n_sigma, self.nx, device=self.device)
        L = self._safe_cholesky(P_pred)
        sigma[:, 0] = x_pred
        for i in range(self.nx):
            sigma[:, i+1] = x_pred + self.gamma * L[:, :, i]
            sigma[:, self.nx+i+1] = x_pred - self.gamma * L[:, :, i]
            
        z_sigma = h_fn(sigma.reshape(-1, self.nx)).reshape(B, self.n_sigma, self.nz)
        z_hat = (self.Wm.view(1, -1, 1) * z_sigma).sum(1)
        y = z - z_hat
        
        S = torch.zeros(B, self.nz, self.nz, device=self.device)
        C = torch.zeros(B, self.nx, self.nz, device=self.device)
        z_diff = z_sigma - z_hat.unsqueeze(1)
        x_diff = sigma - x_pred.unsqueeze(1)
        
        for i in range(self.n_sigma):
            S += self.Wc[i] * (z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
            C += self.Wc[i] * (x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
        S += self.R.expand(B,-1,-1) + 1e-4 * torch.eye(self.nz, device=self.device)
        
        K = C @ torch.linalg.inv(S) # or pinv
        x_post = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_post = P_pred - K @ S @ K.transpose(-1, -2)
        return x_post, P_post, y, S

def nll_loss_stable(y, S):
    try:
        L = torch.linalg.cholesky(S)
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-6), dim=-1)
        z = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False).squeeze(-1)
        mahal = torch.sum(z ** 2, dim=-1)
        return torch.mean(torch.clamp(log_det, -20, 20) + torch.clamp(mahal, 0, 100))
    except: return torch.tensor(20.0, device=y.device)

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_unsupervised_v8(z_train, R, h_fn, x0, P0, nx, nz, epochs=1000, lr=1e-3, device='cpu', checkpoint_path='neural_kf_v8.pth', tbptt_len=5, verbose=True):
    device = torch.device(device)
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    scaler = DataScaler(z_train, device)
    
    if verbose:
        print(f"[Train] Scaled v8 | Mean: {scaler.mean.cpu().numpy()}")

    x0_phys = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_phys = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    mean_net = MeanNetStructured(nx, hidden=64).to(device)
    cov_net = CovNetSimple(nx, init_std=1.0).to(device)
    ukf = UKFCore(nx, nz, R, device)
    
    optimizer = optim.Adam(list(mean_net.parameters()) + list(cov_net.parameters()), lr=lr)
    history = {'loss': [], 'best_epoch': 0}
    best_loss = float('inf')
    
    for ep in range(epochs):
        mean_net.train(); cov_net.train()
        T = min(5 + ep // 10, z_train.shape[1])
        
        batch_idx = torch.randperm(z_train.shape[0])[:min(512, z_train.shape[0])]
        z_batch = z_train[batch_idx]
        
        # Init Norm State
        x_norm = scaler.normalize(x0_phys.unsqueeze(0).expand(len(batch_idx), -1)).clone()
        P_norm = scaler.scale_P_down(P0_phys.unsqueeze(0).expand(len(batch_idx), -1, -1)).clone()
        
        total_loss, n_seg = 0.0, 0
        
        for seg_start in range(0, T, tbptt_len):
            optimizer.zero_grad()
            x_norm, P_norm = x_norm.detach(), P_norm.detach()
            seg_loss, valid = torch.tensor(0.0, device=device, requires_grad=True), 0
            
            for t in range(seg_start, min(seg_start + tbptt_len, T)):
                # 1. Norm Predict
                x_pred_norm = mean_net(x_norm)
                P_pred_norm = cov_net(P_norm)
                # 2. Denorm Update
                x_phys, P_phys, y, S = ukf.update(scaler.denormalize(x_pred_norm), scaler.scale_P_up(P_pred_norm), z_batch[:, t], h_fn)
                # 3. Loss
                loss = nll_loss_stable(y, S)
                if not torch.isnan(loss): seg_loss = seg_loss + loss; valid += 1
                # 4. Renorm
                x_norm = scaler.normalize(x_phys)
                P_norm = scaler.scale_P_down(P_phys)
            
            if valid > 0:
                avg_seg_loss = seg_loss / valid
                # FIX: Check if we have a valid gradient path before backward
                if avg_seg_loss.requires_grad and avg_seg_loss.grad_fn is not None:
                    avg_seg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(mean_net.parameters(), 5.0)
                    optimizer.step()
                    total_loss += avg_seg_loss.item(); n_seg += 1
                else:
                    # Logic: If loss is constant (fail-safe trigger), skip update
                    if verbose and ep % 50 == 0 and n_seg == 0:
                        print(f"Warning: Ep {ep} segment math failed (detached graph). Skipping.")

        if n_seg > 0:
            avg_loss = total_loss / n_seg
            history['loss'].append(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({'mean': mean_net.state_dict(), 'cov': cov_net.state_dict(), 'nx': nx, 
                           'scaler_mean': scaler.mean, 'scaler_std': scaler.std}, checkpoint_path)
            if verbose and ep % 50 == 0: print(f"Ep {ep} | Loss: {avg_loss:.4f}")

    # Load Best & Wrap
    ckpt = torch.load(checkpoint_path, map_location=device)
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    scaler.mean, scaler.std = ckpt['scaler_mean'], ckpt['scaler_std']
    
    # WRAP NETWORKS so test script sees "Physical In -> Physical Out"
    wrapped_mean = ScaledMeanNetWrapper(mean_net, scaler)
    wrapped_cov = ScaledCovNetWrapper(cov_net, scaler)
    
    return wrapped_mean, wrapped_cov, history

# =============================================================================
# INFERENCE CLASS
# =============================================================================

class NeuralKFv8:
    def __init__(self, mean_net, cov_net, R, device):
        self.mean_net = mean_net
        self.cov_net = cov_net
        self.ukf = UKFCore(mean_net.nx, R.shape[0], R, device)
        self.device = device
    
    def run(self, z_seq, x0, P0, h_fn):
        self.mean_net.eval()
        self.cov_net.eval()
        
        # Prepare Data
        z_seq = torch.as_tensor(z_seq, dtype=torch.float32, device=self.device)
        if z_seq.dim() == 2: z_seq = z_seq.unsqueeze(0)
        B, T, _ = z_seq.shape
        
        x = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        if x.dim() == 1: x = x.unsqueeze(0).expand(B, -1)
        
        P = torch.as_tensor(P0, dtype=torch.float32, device=self.device)
        if P.dim() == 2: P = P.unsqueeze(0).expand(B, -1, -1)
        
        states, covs = [], []
        
        with torch.no_grad():
            for t in range(T):
                # 1. Prediction (Wrappers handle scaling internally)
                x_pred = self.mean_net(x)
                P_pred = self.cov_net(P)
                
                # 2. Update (Standard UKF)
                x, P, _, _ = self.ukf.update(x_pred, P_pred, z_seq[:, t], h_fn)
                
                states.append(x.cpu().numpy())
                covs.append(P.cpu().numpy())
        
        # Format output as (Batch, Time, Dims)
        return np.array(states).transpose(1, 0, 2), np.array(covs).transpose(1, 0, 2, 3)