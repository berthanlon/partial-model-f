"""
Neural Kalman Filter v19.1 - STABILIZED DESCENT & COMPLETE

1. Initialization:
   - MeanNet: Exact Zero (Start as Identity).
   - CovNet: Medium Stiffness (Softplus(-4.0) -> ~2m uncertainty).
     *Prevents the "Valley of Death" gradient explosion.*

2. Optimization:
   - LR: 1e-3 (Prevent overshoot).
   - Clip: 0.5 (Prevent jumps).
   
3. Curriculum:
   - 100 Epochs Linear Only -> Then MLP -> Then Covariance.
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
# 1. MEAN NET (Linear Shortcut)
# =============================================================================
class MeanNet(nn.Module):
    def __init__(self, nx, hidden=64):
        super().__init__()
        self.nx = nx
        
        # A. Fast Lane (Linear Physics)
        self.linear = nn.Linear(nx, nx, bias=False)
        
        # B. Slow Lane (Non-linear Residuals)
        self.mlp = nn.Sequential(
            nn.Linear(nx, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, nx)
        )
        
        # --- INITIALIZATION ---
        # 1. Init Linear to EXACT ZERO.
        #    Reason: Even small noise * 100.0 scale = huge physical error.
        #    We rely on the gradient from the loss to push it from zero.
        self.linear.weight.data.zero_()
        
        # 2. Init MLP to Zero
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, x):
        return x + self.linear(x) + self.mlp(x)

# =============================================================================
# 2. COVARIANCE NET (Goldilocks Init)
# =============================================================================
class CovNetFromP(nn.Module):
    def __init__(self, nx, hidden=128):
        super().__init__()
        self.nx = nx
        
        F_anchor = torch.eye(nx)
        self.register_buffer('F_anchor', F_anchor)
        
        self.n_input = nx + (nx * (nx - 1) // 2)
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
        
        # --- MEDIUM STIFFNESS INIT ---
        # Softplus(-4.0) ~= 0.018
        # Scaled (x100) ~= 1.8 meters uncertainty.
        # This is tight enough to encourage physics, but loose enough to prevent explosion.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.001)
            self.net[-1].bias.zero_()
            self.net[-1].bias[:self.n_diag].fill_(-4.0)

    def extract_features(self, P):
        diags = torch.diagonal(P, dim1=1, dim2=2)
        log_diags = torch.log(torch.clamp(diags, min=1e-8))
        lower = P[:, self.tril_idx[0], self.tril_idx[1]]
        return torch.cat([log_diags, lower], dim=1)

    def forward(self, P_old):
        B = P_old.shape[0]
        
        P_anchor = self.F_anchor @ P_old @ self.F_anchor.T
        
        features = self.extract_features(P_old)
        out = self.net(features)
        
        raw_diag = out[:, :self.n_diag]
        diag_vals = F.softplus(raw_diag) + 1e-8
        lower_vals = out[:, self.n_diag:]
        
        L = torch.zeros(B, self.nx, self.nx, device=P_old.device)
        L.as_strided((B, self.nx), (self.nx*self.nx, self.nx + 1)).copy_(diag_vals)
        L[:, self.tril_idx[0], self.tril_idx[1]] = lower_vals
        
        Q = L @ L.transpose(1, 2)
        
        return P_anchor + Q

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
        try: return torch.linalg.cholesky(P + 1e-5 * torch.eye(self.nx, device=self.device))
        except:
            v, e = torch.linalg.eigh(P + 1e-5 * torch.eye(self.nx, device=self.device))
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
        
        S = self.R.clone()
        C = torch.zeros(x_pred.shape[0], self.nx, self.nz, device=self.device)
        
        for i in range(self.n_sigma):
            wd = self.Wc[i].view(1, 1, 1)
            S = S + wd * (z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
            C = C + wd * (x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
            
        try: K = torch.linalg.solve(S, C.transpose(1, 2)).transpose(1, 2)
        except: K = C @ torch.linalg.pinv(S)
            
        y = z - z_hat
        x_new = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_new = P_pred - K @ S @ K.transpose(-1, -2)
        return x_new, P_new, y, S

# =============================================================================
# WRAPPER (Restored)
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
def train_v10(z_train, R, h_fn, x0, P0, nx, nz, epochs=400, lr=1e-3, device='cpu', **kwargs):
    device = torch.device(device)
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    
    scaler = StateScaler(100.0, 10.0, device)
    
    mean_net = MeanNet(nx).to(device)
    cov_net = CovNetFromP(nx).to(device)
    ukf = UKFCore(nx, nz, R, device)
    
    # Optimizer: Uniform 1e-3
    opt_linear = optim.Adam(mean_net.linear.parameters(), lr=1e-3)
    opt_mlp    = optim.Adam(mean_net.mlp.parameters(), lr=1e-3)
    opt_cov    = optim.Adam(cov_net.parameters(), lr=1e-3)
    
    print(f"[Train] v19 Stabilized Descent | Medium Stiffness")
    
    history = {'loss': [], 'best_epoch': 0}
    best_loss = float('inf')
    
    PHASE_2 = 100 # Unfreeze MLP
    PHASE_3 = 200 # Unfreeze Cov
    
    for ep in range(epochs):
        train_linear = True
        train_mlp    = (ep >= PHASE_2)
        train_cov    = (ep >= PHASE_3)
        
        if ep == PHASE_2: print(">>> Phase 2: Unfreezing MLP <<<")
        if ep == PHASE_3: print(">>> Phase 3: Unfreezing Covariance <<<")
            
        mean_net.train()
        cov_net.train() if train_cov else cov_net.eval()
        
        opt_linear.zero_grad()
        if train_mlp: opt_mlp.zero_grad()
        if train_cov: opt_cov.zero_grad()
        
        B = z_train.shape[0]
        x = torch.as_tensor(x0, device=device).unsqueeze(0).expand(B, -1)
        P = torch.as_tensor(P0, device=device).unsqueeze(0).expand(B, -1, -1)
        
        total_loss = 0
        loss_chunk = 0
        tbptt_len = 10
        
        for t in range(z_train.shape[1]):
            x_norm = scaler.normalize(x)
            P_norm = scaler.scale_cov(P)
            
            x_pred_norm = mean_net(x_norm)
            P_pred_norm = cov_net(P_norm)
            
            x_pred = scaler.denormalize(x_pred_norm)
            P_pred = scaler.unscale_cov(P_pred_norm)
            
            x, P, y, S = ukf.update(x_pred, P_pred, z_train[:, t], h_fn)
            
            try:
                L_S = torch.linalg.cholesky(S)
                log_det = 2 * torch.sum(torch.log(torch.diagonal(L_S, dim1=-2, dim2=-1)), dim=-1)
                sol = torch.linalg.solve(L_S, y.unsqueeze(-1)).squeeze(-1)
                mahal = torch.sum(sol**2, dim=-1)
            except:
                mahal = torch.sum(y**2, dim=-1)
                log_det = torch.zeros_like(mahal) + 5.0
                
            loss_t = torch.mean(log_det + mahal)
            loss_chunk += loss_t
            
            if (t + 1) % tbptt_len == 0:
                avg_loss = loss_chunk / tbptt_len
                avg_loss.backward()
                
                # Strict Clipping (0.5) to prevent jumping
                torch.nn.utils.clip_grad_norm_(mean_net.linear.parameters(), 0.5)
                opt_linear.step()
                opt_linear.zero_grad()
                
                if train_mlp:
                    torch.nn.utils.clip_grad_norm_(mean_net.mlp.parameters(), 0.5)
                    opt_mlp.step()
                    opt_mlp.zero_grad()
                
                if train_cov:
                    torch.nn.utils.clip_grad_norm_(cov_net.parameters(), 0.5)
                    opt_cov.step()
                    opt_cov.zero_grad()
                
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