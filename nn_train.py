"""
Neural Kalman Filter v10 - STRICT MATH + STABILIZED TRAINING

1. Prediction Mean: x_k|k-1 = m_theta(x_k-1|k-1)
2. Prediction Cov: P_k|k-1 = C_theta(P_k-1|k-1)
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
# 1. MEAN MAPPING m_theta(x)
# =============================================================================
class MeanNet(nn.Module):
    def __init__(self, nx, hidden=64, dt_norm=0.05):
        super().__init__()
        self.nx = nx
        self.dt_norm = dt_norm
        
        self.net = nn.Sequential(
            nn.Linear(nx, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, nx)
        )
        self._init_physics()

    def _init_physics(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.zeros_(m.bias)
        
        with torch.no_grad():
            l1 = self.net[0]
            l_out = self.net[-1]
            l1.weight.zero_()
            l1.weight[0, 1] = 1.0 
            l1.weight[1, 3] = 1.0 
            l_out.weight.zero_()
            l_out.weight[0, 0] = self.dt_norm 
            l_out.weight[2, 1] = self.dt_norm 

    def forward(self, x):
        return x + self.net(x)

# =============================================================================
# 2. COVARIANCE MAPPING C_theta(P)
# =============================================================================
class CovNetFromP(nn.Module):
    def __init__(self, nx, hidden=128):
        super().__init__()
        self.nx = nx
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
        
        with torch.no_grad():
            self.net[-1].weight.fill_(0.01)
            self.net[-1].bias.zero_()
            self.net[-1].bias[:self.n_diag].fill_(0.54)

    def extract_features(self, P):
        diags = torch.diagonal(P, dim1=1, dim2=2)
        log_diags = torch.log(torch.clamp(diags, min=1e-6))
        lower = P[:, self.tril_idx[0], self.tril_idx[1]]
        return torch.cat([log_diags, lower], dim=1)

    def forward(self, P_old):
        features = self.extract_features(P_old)
        out = self.net(features)
        
        raw_diag = out[:, :self.n_diag]
        diag_vals = F.softplus(raw_diag) + 1e-6
        lower_vals = out[:, self.n_diag:]
        
        B = P_old.shape[0]
        L = torch.zeros(B, self.nx, self.nx, device=P_old.device)
        L.as_strided((B, self.nx), (self.nx*self.nx, self.nx + 1)).copy_(diag_vals)
        L[:, self.tril_idx[0], self.tril_idx[1]] = lower_vals
        
        return L @ L.transpose(1, 2)

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
        for i in range(self.n_sigma):
            term = z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2)
            S = S + self.Wc[i] * term
        
        C = torch.zeros(x_pred.shape[0], self.nx, self.nz, device=self.device)
        for i in range(self.n_sigma):
            term = x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2)
            C = C + self.Wc[i] * term
            
        try: K = torch.linalg.solve(S, C.transpose(1, 2)).transpose(1, 2)
        except: K = C @ torch.linalg.pinv(S)
            
        y = z - z_hat
        x_new = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_new = P_pred - K @ S @ K.transpose(-1, -2)
        return x_new, P_new, y, S

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_v10(z_train, R, h_fn, x0, P0, nx, nz, epochs=200, lr=1e-3, device='cpu', **kwargs):
    device = torch.device(device)
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    
    pos_scale, vel_scale = 100.0, 10.0
    dt_norm = 0.5 * vel_scale / pos_scale
    
    mean_net = MeanNet(nx, dt_norm=dt_norm).to(device)
    cov_net = CovNetFromP(nx).to(device)
    scaler = StateScaler(pos_scale, vel_scale, device)
    ukf = UKFCore(nx, nz, R, device)
    
    optimizer = optim.Adam([
        {'params': mean_net.parameters(), 'lr': 1e-4}, 
        {'params': cov_net.parameters(), 'lr': 1e-3}
    ])
    
    # NEW: Learning Rate Scheduler
    # Reduces LR by factor of 0.5 if loss doesn't improve for 20 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    print(f"[Train] v10 Stabilized | Scheduler Active")
    
    history = {'loss': [], 'best_epoch': 0}
    best_loss = float('inf')
    
    for ep in range(epochs):
        mean_net.train(); cov_net.train()
        optimizer.zero_grad()
        
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
                log_det = torch.zeros_like(mahal)
                
            loss_chunk += torch.mean(log_det + mahal)
            
            if (t + 1) % tbptt_len == 0:
                avg_loss = loss_chunk / tbptt_len
                avg_loss.backward()
                # NEW: Tighter Clipping
                torch.nn.utils.clip_grad_norm_(mean_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(cov_net.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                x = x.detach(); P = P.detach()
                total_loss += avg_loss.item()
                loss_chunk = 0
        
        # Step Scheduler
        scheduler.step(total_loss)

        if ep % 20 == 0:
            print(f"Ep {ep} | Loss: {total_loss:.4f}")
        
        history['loss'].append(total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            history['best_epoch'] = ep

    return mean_net, cov_net, scaler, history