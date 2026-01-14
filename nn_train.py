# nn_train_v7.py
"""
FULLY UNSUPERVISED Neural Kalman Filter - v7

Key improvements over v6:
1. Structured state network: separate position and velocity processing
2. Explicit velocity integration bias (helps learn x += v*dt)
3. Better gradient clipping and stability
4. Detached covariance input to prevent gradient explosion through P

The network must learn:
- Position update: x_next = x + f(v) where f(v) ≈ v*dt
- Velocity update: v_next = g(v) where g(v) ≈ v (persistence)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func


# =============================================================================
# NETWORKS - STRUCTURED FOR DYNAMICS LEARNING
# =============================================================================

class MeanNetStructured(nn.Module):
    """
    Structured state prediction that biases toward constant-velocity dynamics.
    
    State: [x, vx, y, vy]
    
    Architecture encourages:
    - x_next = x + velocity_contribution(vx)
    - vx_next = vx + small_correction
    - Same for y, vy
    
    The network learns the dt implicitly through the velocity contribution.
    """
    def __init__(self, nx, hidden=64):
        super().__init__()
        self.nx = nx
        
        # Velocity-to-position network: learns v*dt relationship
        # Input: velocity (vx or vy), Output: position delta
        self.vel_to_pos = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        # Velocity update network: learns velocity persistence/change
        # Input: velocity, Output: velocity delta (should be ~0 for CV model)
        self.vel_update = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        # Initialize vel_to_pos to approximate identity (v*1 = v)
        # This gives a starting point close to dt=1
        self._init_vel_to_pos()
        
        # Initialize vel_update to output ~0 (velocity persists)
        self._init_vel_update()
    
    def _init_vel_to_pos(self):
        """Initialize to approximate f(v) ≈ v (identity-ish)."""
        for m in self.vel_to_pos:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # Last layer: small weights, will learn the dt scaling
        nn.init.normal_(self.vel_to_pos[-1].weight, mean=0.5, std=0.1)
        nn.init.zeros_(self.vel_to_pos[-1].bias)
    
    def _init_vel_update(self):
        """Initialize to output ~0 (velocity doesn't change much)."""
        for m in self.vel_update:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, 4) = [x, vx, y, vy]
        """
        B = x.shape[0]
        
        pos_x, vel_x = x[:, 0:1], x[:, 1:2]
        pos_y, vel_y = x[:, 2:3], x[:, 3:4]
        
        # Position update: x_new = x + f(vx)
        # The network learns f(v) ≈ v * dt
        dx = self.vel_to_pos(vel_x)
        dy = self.vel_to_pos(vel_y)
        
        pos_x_new = pos_x + dx
        pos_y_new = pos_y + dy
        
        # Velocity update: v_new = v + g(v)
        # The network learns g(v) ≈ 0 for constant velocity
        # Clamp the velocity delta to prevent runaway
        dvx = torch.clamp(self.vel_update(vel_x), -0.5, 0.5)
        dvy = torch.clamp(self.vel_update(vel_y), -0.5, 0.5)
        
        vel_x_new = vel_x + dvx
        vel_y_new = vel_y + dvy
        
        return torch.cat([pos_x_new, vel_x_new, pos_y_new, vel_y_new], dim=1)


class CovNetSimple(nn.Module):
    """
    Simplified covariance prediction.
    
    Instead of complex learning, use a simpler approach:
    - Learn a base Cholesky factor
    - Scale it based on input covariance diagonal
    
    This is more stable and still allows adaptation.
    """
    def __init__(self, nx, init_std=1.0):
        super().__init__()
        self.nx = nx
        
        # Learnable Cholesky diagonal (log scale for positivity)
        self.log_L_diag = nn.Parameter(torch.zeros(nx))
        
        # Learnable lower triangle
        n_lower = nx * (nx - 1) // 2
        self.L_lower = nn.Parameter(torch.zeros(n_lower))
        
        # Small network to adapt based on input
        self.adapt_net = nn.Sequential(
            nn.Linear(nx, 32),
            nn.Tanh(),
            nn.Linear(32, nx)
        )
        
        # Initialize
        nn.init.constant_(self.log_L_diag, np.log(init_std))
        nn.init.zeros_(self.L_lower)
        
        for m in self.adapt_net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        # Indices
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
    
    def forward(self, P_prev):
        B = P_prev.shape[0]
        device = P_prev.device
        
        # Get diagonal of input (detached to prevent gradient explosion)
        diag_in = torch.diagonal(P_prev.detach(), dim1=-2, dim2=-1)
        log_diag_in = torch.log(diag_in.clamp(min=1e-6))
        
        # Adaptive scaling based on input
        adapt = self.adapt_net(log_diag_in)  # (B, nx)
        
        # Build L diagonal: base + adaptation
        L_diag = torch.exp(self.log_L_diag + 0.1 * adapt)
        L_diag = torch.clamp(L_diag, min=0.01, max=100.0)
        
        # Build L matrix
        L = torch.zeros(B, self.nx, self.nx, device=device)
        L[:, range(self.nx), range(self.nx)] = L_diag
        
        if self.L_lower.numel() > 0:
            # Lower triangle: scaled by diagonal
            scale = torch.sqrt(L_diag[:, self.tril_idx[0]] * L_diag[:, self.tril_idx[1]])
            L_low = torch.tanh(self.L_lower) * 0.3 * scale
            L[:, self.tril_idx[0], self.tril_idx[1]] = L_low
        
        # P = L @ L^T
        P = L @ L.transpose(-1, -2)
        
        return P


# =============================================================================
# UKF CORE (same as before)
# =============================================================================

class UKFCore:
    def __init__(self, nx, nz, R, device, alpha=0.001, beta=2.0, kappa=0.0):
        self.nx = nx
        self.nz = nz
        self.device = device
        
        self.R = torch.as_tensor(R, dtype=torch.float32, device=device)
        if self.R.dim() == 2:
            self.R = self.R.unsqueeze(0)
        
        lmbda = alpha**2 * (nx + kappa) - nx
        self.gamma = np.sqrt(nx + lmbda)
        
        n_sigma = 2 * nx + 1
        Wm = torch.zeros(n_sigma, device=device)
        Wc = torch.zeros(n_sigma, device=device)
        Wm[0] = lmbda / (nx + lmbda)
        Wc[0] = Wm[0] + (1 - alpha**2 + beta)
        Wm[1:] = 0.5 / (nx + lmbda)
        Wc[1:] = Wm[1:]
        
        self.Wm = Wm
        self.Wc = Wc
        self.n_sigma = n_sigma
    
    def _safe_cholesky(self, P):
        P_reg = P + 1e-4 * torch.eye(self.nx, device=self.device)
        try:
            return torch.linalg.cholesky(P_reg)
        except:
            eigvals, eigvecs = torch.linalg.eigh(P_reg)
            eigvals = torch.clamp(eigvals, min=1e-4)
            return eigvecs @ torch.diag_embed(torch.sqrt(eigvals))
    
    def _sigma_points(self, x, P):
        B = x.shape[0]
        L = self._safe_cholesky(P)
        
        sigma = torch.zeros(B, self.n_sigma, self.nx, device=self.device)
        sigma[:, 0] = x
        
        scaled_L = self.gamma * L
        for i in range(self.nx):
            sigma[:, i+1] = x + scaled_L[:, :, i]
            sigma[:, self.nx+i+1] = x - scaled_L[:, :, i]
        
        return sigma
    
    def update(self, x_pred, P_pred, z, h_fn):
        B = x_pred.shape[0]
        
        sigma = self._sigma_points(x_pred, P_pred)
        
        sigma_flat = sigma.reshape(B * self.n_sigma, self.nx)
        z_sigma_flat = h_fn(sigma_flat)
        z_sigma = z_sigma_flat.reshape(B, self.n_sigma, self.nz)
        
        z_hat = (self.Wm.view(1, -1, 1) * z_sigma).sum(dim=1)
        
        z_diff = z_sigma - z_hat.unsqueeze(1)
        S = torch.zeros(B, self.nz, self.nz, device=self.device)
        for i in range(self.n_sigma):
            S = S + self.Wc[i] * (z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
        S = S + self.R.expand(B, -1, -1)
        S = 0.5 * (S + S.transpose(-1, -2))
        S = S + 1e-4 * torch.eye(self.nz, device=self.device)
        
        x_diff = sigma - x_pred.unsqueeze(1)
        C = torch.zeros(B, self.nx, self.nz, device=self.device)
        for i in range(self.n_sigma):
            C = C + self.Wc[i] * (x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
        
        S_inv = torch.linalg.inv(S)
        K = C @ S_inv
        
        y = z - z_hat
        
        x_post = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_post = P_pred - K @ S @ K.transpose(-1, -2)
        P_post = 0.5 * (P_post + P_post.transpose(-1, -2))
        
        # Project to SPD
        eigvals, eigvecs = torch.linalg.eigh(P_post)
        eigvals = torch.clamp(eigvals, min=1e-4)
        P_post = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        
        return x_post, P_post, y, S


# =============================================================================
# LOSS
# =============================================================================

def nll_loss(y, S):
    """NLL with stability."""
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clamp(eigvals, min=1e-6, max=1e6)
    
    log_det = torch.sum(torch.log(eigvals), dim=-1)
    log_det = torch.clamp(log_det, -100, 100)
    
    y_rot = (eigvecs.transpose(-1, -2) @ y.unsqueeze(-1)).squeeze(-1)
    mahal = torch.sum(y_rot**2 / eigvals, dim=-1)
    mahal = torch.clamp(mahal, max=100.0)
    
    return torch.mean(log_det + mahal)


# =============================================================================
# TRAINING
# =============================================================================

def train_unsupervised_v7(
    z_train, R, h_fn,
    x0, P0,
    nx, nz,
    epochs=500,
    lr=1e-3,
    device='cpu',
    checkpoint_path='neural_kf_v7.pth',
    verbose=True
):
    """
    Train v7 with structured dynamics learning.
    """
    device = torch.device(device)
    
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    B, T_max, _ = z_train.shape
    
    if verbose:
        print(f"[Train] v7 - Structured Dynamics Learning")
        print(f"[Train] B={B}, T_max={T_max}, nx={nx}, nz={nz}")
    
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_t = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    # Networks
    mean_net = MeanNetStructured(nx, hidden=64).to(device)
    cov_net = CovNetSimple(nx, init_std=1.0).to(device)
    
    # UKF
    ukf = UKFCore(nx, nz, R, device)
    
    # Separate optimizers for better control
    opt_mean = optim.Adam(mean_net.parameters(), lr=lr)
    opt_cov = optim.Adam(cov_net.parameters(), lr=lr * 0.5)
    
    # Schedulers
    sched_mean = optim.lr_scheduler.ReduceLROnPlateau(opt_mean, factor=0.5, patience=50, min_lr=1e-6)
    sched_cov = optim.lr_scheduler.ReduceLROnPlateau(opt_cov, factor=0.5, patience=50, min_lr=1e-6)
    
    history = {'loss': [], 'best_epoch': 0, 'T_curriculum': []}
    best_loss = float('inf')
    
    for ep in range(epochs):
        mean_net.train()
        cov_net.train()
        opt_mean.zero_grad()
        opt_cov.zero_grad()
        
        # Curriculum: start short
        T = min(3 + ep // 30, T_max)
        history['T_curriculum'].append(T)
        
        # Random start for variety
        if T < T_max:
            start = np.random.randint(0, T_max - T + 1)
        else:
            start = 0
        
        # Initialize
        x = x0_t.unsqueeze(0).expand(B, -1).clone()
        P = P0_t.unsqueeze(0).expand(B, -1, -1).clone()
        
        total_loss = 0.0
        valid = 0
        
        for t in range(T):
            x_pred = mean_net(x)
            P_pred = cov_net(P)
            
            x, P, y, S = ukf.update(x_pred, P_pred, z_train[:, start + t], h_fn)
            
            loss_t = nll_loss(y, S)
            
            if not (torch.isnan(loss_t) or torch.isinf(loss_t)):
                total_loss = total_loss + loss_t
                valid += 1
        
        if valid == 0:
            if verbose:
                print(f"[Train] Ep {ep}: all NaN, skipping")
            continue
        
        loss = total_loss / valid
        loss.backward()
        
        # Separate gradient clipping
        gn_mean = torch.nn.utils.clip_grad_norm_(mean_net.parameters(), 5.0)
        gn_cov = torch.nn.utils.clip_grad_norm_(cov_net.parameters(), 5.0)
        
        if gn_mean > 50 or gn_cov > 50:
            if verbose and ep % 100 == 0:
                print(f"[Train] Ep {ep}: grad explosion (mean={gn_mean:.1f}, cov={gn_cov:.1f})")
            continue
        
        opt_mean.step()
        opt_cov.step()
        
        loss_val = loss.item()
        history['loss'].append(loss_val)
        
        sched_mean.step(loss_val)
        sched_cov.step(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            history['best_epoch'] = ep
            torch.save({
                'mean': mean_net.state_dict(),
                'cov': cov_net.state_dict(),
                'nx': nx,
            }, checkpoint_path)
        
        if verbose and ep % 100 == 0:
            print(f"Ep {ep:4d} | Loss: {loss_val:.4f} | Best: {best_loss:.4f} @ {history['best_epoch']} | T={T}")
    
    # Load best
    ckpt = torch.load(checkpoint_path, map_location=device)
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    if verbose:
        print(f"\n[Train] Done. Best: {best_loss:.4f} @ epoch {history['best_epoch']}")
    
    return mean_net, cov_net, history


# =============================================================================
# INFERENCE
# =============================================================================

class NeuralKFv7:
    def __init__(self, mean_net, cov_net, R, device):
        self.mean_net = mean_net.to(device)
        self.cov_net = cov_net.to(device)
        self.device = device
        self.nx = mean_net.nx
        self.nz = R.shape[0]
        self.ukf = UKFCore(self.nx, self.nz, R, device)
    
    def run(self, z_seq, x0, P0, h_fn):
        self.mean_net.eval()
        self.cov_net.eval()
        
        z_seq = torch.as_tensor(z_seq, dtype=torch.float32, device=self.device)
        if z_seq.dim() == 2:
            z_seq = z_seq.unsqueeze(0)
        
        B, T, _ = z_seq.shape
        
        x = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)
        
        P = torch.as_tensor(P0, dtype=torch.float32, device=self.device)
        if P.dim() == 2:
            P = P.unsqueeze(0).expand(B, -1, -1)
        
        states, covs = [], []
        
        with torch.no_grad():
            for t in range(T):
                x_pred = self.mean_net(x)
                P_pred = self.cov_net(P)
                x, P, _, _ = self.ukf.update(x_pred, P_pred, z_seq[:, t], h_fn)
                states.append(x.cpu().numpy())
                covs.append(P.cpu().numpy())
        
        return np.array(states).transpose(1, 0, 2), np.array(covs).transpose(1, 0, 2, 3)


def load_v7(path, R, device):
    ckpt = torch.load(path, map_location=device)
    nx = ckpt['nx']
    
    mean_net = MeanNetStructured(nx).to(device)
    cov_net = CovNetSimple(nx).to(device)
    
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    return NeuralKFv7(mean_net, cov_net, R, device)