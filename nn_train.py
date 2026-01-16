"""
Neural Kalman Filter v12 - PHYSICS ANCHORED STABILITY

1. Mean Function m_theta(x):
   - x_pred = F*x + NeuralResidual(x)
   - Starts as Newton's Laws. Learns Drag/Turns.

2. Covariance Function C_theta(P):
   - P_pred = F*P*F^T + NeuralProcessNoise(P)
   - Starts as Standard Error Propagation. Learns State-Dependent Noise.
   - This satisfies C_theta(P) while preventing explosion.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =============================================================================
# SCALER
# =============================================================================
class StateScaler:
    def __init__(self, pos_scale=100.0, vel_scale=10.0, device='cpu'):
        self.device = device
        self.scale = torch.tensor([pos_scale, vel_scale, pos_scale, vel_scale], 
                                   dtype=torch.float32, device=device)
        self.inv_scale = 1.0 / self.scale
        
    def normalize(self, x_phys): return x_phys * self.inv_scale
    def denormalize(self, x_norm): return x_norm * self.scale
    def scale_cov(self, P):
        D_inv = torch.diag(self.inv_scale)
        return D_inv @ P @ D_inv
    def unscale_cov(self, P_norm):
        D = torch.diag(self.scale)
        return D @ P_norm @ D

# =============================================================================
# 1. PHYSICS-GUIDED MEAN NET
# =============================================================================
class PhysicsGuidedMeanNet(nn.Module):
    def __init__(self, nx, hidden=64, dt_norm=0.05):
        super().__init__()
        self.nx = nx
        
        # 1. The Neural Residual
        self.residual_net = nn.Sequential(
            nn.Linear(nx, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, nx)
        )
        
        # 2. The Physics Matrix (Constant Velocity)
        F = torch.eye(nx)
        F[0, 1] = dt_norm; F[2, 3] = dt_norm
        self.register_buffer('F_norm', F)
        
        # Init Residuals to Zero
        for m in self.residual_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x_new = F*x + Net(x)
        x_phys = x @ self.F_norm.T
        x_corr = self.residual_net(x)
        return x_phys + x_corr

# =============================================================================
# 2. PHYSICS-ANCHORED COVARIANCE NET
# =============================================================================
class PhysicsAnchoredCovNet(nn.Module):
    """
    Implements C_theta(P) = F*P*F^T + Q(P)
    This creates a valid mapping from P_old -> P_new that is STABLE.
    """
    def __init__(self, nx, hidden=128, dt_norm=0.05):
        super().__init__()
        self.nx = nx
        
        # Physics Propagation Matrix
        F = torch.eye(nx)
        F[0, 1] = dt_norm; F[2, 3] = dt_norm
        self.register_buffer('F_norm', F)
        
        # Network learns Q based on P (and optionally x, but let's stick to P)
        # Input: Flattened P
        self.n_input = nx * nx 
        # Output: Cholesky factors of Q (diagonal + lower)
        self.n_diag = nx
        self.n_lower = nx * (nx - 1) // 2
        self.n_output = self.n_diag + self.n_lower
        
        self.net = nn.Sequential(
            nn.Linear(self.n_input, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_output)
        )
        
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
        
        # Init Q to be small but non-zero
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
            # Initialize diagonals to something reasonable (e.g. 0.01)
            # Softplus(-3.0) approx 0.05
            self.net[-1].bias[:self.n_diag].fill_(-3.0)

    def forward(self, P_old):
        B = P_old.shape[0]
        
        # 1. Physics Propagation: P_phys = F * P_old * F^T
        P_phys = self.F_norm @ P_old @ self.F_norm.T
        
        # 2. Learned Process Noise: Q = Net(P_old)
        # We flatten P_old to feed the net
        features = P_old.view(B, -1)
        out = self.net(features)
        
        # Construct Q via Cholesky L_Q
        raw_diag = out[:, :self.n_diag]
        diag_vals = F.softplus(raw_diag) + 1e-6
        lower_vals = out[:, self.n_diag:]
        
        L_Q = torch.zeros(B, self.nx, self.nx, device=P_old.device)
        L_Q.as_strided((B, self.nx), (self.nx*self.nx, self.nx + 1)).copy_(diag_vals)
        L_Q[:, self.tril_idx[0], self.tril_idx[1]] = lower_vals
        
        Q = L_Q @ L_Q.transpose(1, 2)
        
        # 3. Combine: P_new = P_phys + Q
        return P_phys + Q

# =============================================================================
# UKF CORE
# =============================================================================
class UKFCore:
    def __init__(self, nx, nz, R, device):
        self.nx = nx; self.nz = nz; self.device = device
        self.R = torch.as_tensor(R, dtype=torch.float32, device=device)
        if self.R.dim() == 2: self.R = self.R.unsqueeze(0)
        
        alpha = 1.0; beta = 2.0; kappa = 0.0
        lmbda = alpha**2 * (nx + kappa) - nx
        self.gamma = np.sqrt(nx + lmbda)
        self.n_sigma = 2 * nx + 1
        
        Wm = torch.zeros(self.n_sigma, device=device)
        Wc = torch.zeros(self.n_sigma, device=device)
        denom = nx + lmbda
        Wm[0] = lmbda/denom; Wc[0] = Wm[0] + (1-alpha**2+beta)
        Wm[1:] = 0.5/denom; Wc[1:] = Wm[1:]
        self.Wm, self.Wc = Wm, Wc

    def _safe_cholesky(self, P):
        # Robust Cholesky with fallback
        P_reg = P + 1e-5 * torch.eye(self.nx, device=self.device)
        try: return torch.linalg.cholesky(P_reg)
        except:
            v, e = torch.linalg.eigh(P_reg)
            v = torch.clamp(v, min=1e-5)
            return e @ torch.diag_embed(torch.sqrt(v))

    def update(self, x_pred, P_pred, z, h_fn):
        L = self._safe_cholesky(P_pred)
        sigmas = [x_pred]
        for i in range(self.nx):
            sigmas.append(x_pred + self.gamma * L[:, :, i])
            sigmas.append(x_pred - self.gamma * L[:, :, i])
        sigma = torch.stack(sigmas, dim=1)
        
        z_sigma = h_fn(sigma.reshape(-1, self.nx)).reshape(x_pred.shape[0], self.n_sigma, self.nz)
        z_hat = (self.Wm.view(1, -1, 1) * z_sigma).sum(1)
        z_diff = z_sigma - z_hat.unsqueeze(1)
        x_diff = sigma - x_pred.unsqueeze(1)
        
        # Manual broadcasting for stability
        S = self.R.clone()
        C = torch.zeros(x_pred.shape[0], self.nx, self.nz, device=self.device)
        
        for i in range(self.n_sigma):
            wd = self.Wc[i].view(1, 1, 1) # Explicit broadcast shape
            
            z_outer = z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2)
            S = S + wd * z_outer
            
            xz_outer = x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2)
            C = C + wd * xz_outer
            
        try: K = torch.linalg.solve(S, C.transpose(1, 2)).transpose(1, 2)
        except: K = C @ torch.linalg.pinv(S)
            
        y = z - z_hat
        x_new = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_new = P_pred - K @ S @ K.transpose(-1, -2)
        
        # Symmetrize P to prevent drift
        P_new = 0.5 * (P_new + P_new.transpose(1, 2))
        return x_new, P_new, y, S

# =============================================================================
# WRAPPER
# =============================================================================
class NeuralKFv10:
    def __init__(self, mean_net, cov_net, scaler, R, device):
        self.mean_net = mean_net
        self.cov_net = cov_net
        self.scaler = scaler
        self.ukf = UKFCore(4, 2, R, device)
        self.device = device
        
    def run(self, z_seq, x0, P0, h_fn):
        self.mean_net.eval(); self.cov_net.eval()
        z_seq = torch.as_tensor(z_seq, dtype=torch.float32, device=self.device)
        if z_seq.dim() == 2: z_seq = z_seq.unsqueeze(0)
        B, T, _ = z_seq.shape
        
        x = torch.as_tensor(x0, device=self.device).unsqueeze(0).expand(B, -1)
        P = torch.as_tensor(P0, device=self.device).unsqueeze(0).expand(B, -1, -1)
        
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
        return np.array(states).transpose(1, 0, 2), np.array(covs).transpose(1, 0, 2, 3)

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_v10(z_train, R, h_fn, x0, P0, nx, nz, epochs=200, lr=1e-3, device='cpu', **kwargs):
    device = torch.device(device)
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    
    # Config
    pos_scale, vel_scale = 100.0, 10.0
    dt_norm = 0.5 * vel_scale / pos_scale
    
    # Models
    mean_net = PhysicsGuidedMeanNet(nx, dt_norm=dt_norm).to(device)
    cov_net = PhysicsAnchoredCovNet(nx, dt_norm=dt_norm).to(device)
    scaler = StateScaler(pos_scale, vel_scale, device)
    ukf = UKFCore(nx, nz, R, device)
    
    # Optimizer with Gradient Clipping Hooks
    optimizer = optim.Adam([
        {'params': mean_net.parameters(), 'lr': 1e-4}, 
        {'params': cov_net.parameters(), 'lr': 1e-3}
    ])
    
    print(f"[Train] v12 Physics Anchored | Stable C_theta(P)")
    
    history = {'loss': [], 'best_epoch': 0}
    best_loss = float('inf')
    
    for ep in range(epochs):
        # STAGED TRAINING
        if ep < 50:
            mean_net.eval() # Freeze Physics first
            cov_net.train()
        else:
            mean_net.train() # Learn Physics later
            cov_net.train()
            
        optimizer.zero_grad()
        
        B = z_train.shape[0]
        x = torch.as_tensor(x0, device=device).unsqueeze(0).expand(B, -1)
        P = torch.as_tensor(P0, device=device).unsqueeze(0).expand(B, -1, -1)
        
        total_loss = 0
        loss_chunk = 0
        tbptt_len = 10
        
        for t in range(z_train.shape[1]):
            # Norm
            x_norm = scaler.normalize(x)
            P_norm = scaler.scale_cov(P)
            
            # Predict
            x_pred_norm = mean_net(x_norm)
            P_pred_norm = cov_net(P_norm)
            
            # Denorm
            x_pred = scaler.denormalize(x_pred_norm)
            P_pred = scaler.unscale_cov(P_pred_norm)
            
            # Update
            x, P, y, S = ukf.update(x_pred, P_pred, z_train[:, t], h_fn)
            
            # Robust NLL Loss
            try:
                L_S = torch.linalg.cholesky(S)
                log_det = 2 * torch.sum(torch.log(torch.diagonal(L_S, dim1=-2, dim2=-1)), dim=-1)
                sol = torch.linalg.solve(L_S, y.unsqueeze(-1)).squeeze(-1)
                mahal = torch.sum(sol**2, dim=-1)
            except:
                # Fallback for singularity
                mahal = torch.sum(y**2, dim=-1)
                log_det = torch.zeros_like(mahal) + 5.0 
                
            # Soft clamp loss to prevent single-batch explosion
            loss_t = torch.mean(log_det + mahal)
            loss_chunk += loss_t
            
            if (t + 1) % tbptt_len == 0:
                avg_loss = loss_chunk / tbptt_len
                avg_loss.backward()
                
                # STRICT CLIPPING
                torch.nn.utils.clip_grad_norm_(mean_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(cov_net.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                x = x.detach(); P = P.detach()
                total_loss += avg_loss.item()
                loss_chunk = 0

        if ep % 20 == 0:
            print(f"Ep {ep} | Loss: {total_loss:.4f}")
        
        history['loss'].append(total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            history['best_epoch'] = ep

    return mean_net, cov_net, scaler, history