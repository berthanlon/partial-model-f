# nn_train_v6.py
"""
FULLY UNSUPERVISED Neural Kalman Filter - v6

NO KNOWN DYNAMICS (F matrix not used)

The neural networks must learn:
1. State transition dynamics from data
2. Process noise covariance

This is harder and requires:
- More training data
- Longer training
- Careful architecture design
- Curriculum learning (short sequences first)

Key insight: We use an RNN-style approach where the network sees
the previous state and must predict the next state distribution.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# =============================================================================
# NETWORKS - FULLY LEARNED (NO F MATRIX)
# =============================================================================

class MeanNetUnsupervised(nn.Module):
    """
    State prediction network - learns dynamics from scratch.
    
    x_pred = f_nn(x_prev)
    
    Architecture designed to learn constant-velocity-like dynamics:
    - Residual connection (identity + learned delta)
    - Separate processing for position and velocity components
    """
    def __init__(self, nx, hidden=128, layers=3):
        super().__init__()
        self.nx = nx
        
        # Main network
        modules = []
        prev = nx
        for i in range(layers):
            modules.append(nn.Linear(prev, hidden))
            modules.append(nn.LayerNorm(hidden))
            modules.append(nn.ELU())  # ELU for smoother gradients
            prev = hidden
        modules.append(nn.Linear(hidden, nx))
        self.net = nn.Sequential(*modules)
        
        # Initialize for near-identity at start
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # Last layer very small
        nn.init.normal_(self.net[-1].weight, 0, 0.01)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        """
        x: (B, nx) = [x, vx, y, vy]
        returns: x_pred (B, nx)
        """
        # Network predicts the full next state
        # Use residual connection to help learning
        delta = self.net(x)
        
        # Residual: x_next = x_prev + delta
        # This biases toward identity initially, network learns the correction
        return x + delta


class CovNetUnsupervised(nn.Module):
    """
    Covariance prediction via Cholesky - learns Q from scratch.
    
    P_pred = L @ L^T
    
    The network learns what the predicted covariance should be
    based on the previous covariance.
    """
    def __init__(self, nx, hidden=128, layers=3, 
                 min_var=0.01, max_var=1000.0):
        super().__init__()
        self.nx = nx
        self.min_var = min_var
        self.max_var = max_var
        
        self.n_diag = nx
        self.n_lower = nx * (nx - 1) // 2
        self.n_out = self.n_diag + self.n_lower
        
        # Input: diagonal of P in log space + lower triangle
        n_input = nx + self.n_lower
        
        modules = []
        prev = n_input
        for _ in range(layers):
            modules.append(nn.Linear(prev, hidden))
            modules.append(nn.LayerNorm(hidden))
            modules.append(nn.ELU())
            prev = hidden
        modules.append(nn.Linear(hidden, self.n_out))
        self.net = nn.Sequential(*modules)
        
        # Indices
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # Initialize output bias for reasonable starting covariance
        # We want L_diag ~ 1.0 initially, so softplus^{-1}(1) ~ 0.54
        nn.init.constant_(self.net[-1].bias[:self.n_diag], 0.5)
        nn.init.zeros_(self.net[-1].bias[self.n_diag:])
    
    def forward(self, P_prev):
        B = P_prev.shape[0]
        device = P_prev.device
        
        # Extract features from P_prev
        diag = torch.diagonal(P_prev, dim1=-2, dim2=-1)
        log_diag = torch.log(diag.clamp(min=1e-6))
        
        if self.n_lower > 0:
            lower = P_prev[:, self.tril_idx[0], self.tril_idx[1]]
            # Normalize lower by diagonal scale
            scale = torch.sqrt(diag[:, self.tril_idx[0]] * diag[:, self.tril_idx[1]]).clamp(min=1e-6)
            lower_norm = lower / scale
            features = torch.cat([log_diag, lower_norm], dim=-1)
        else:
            features = log_diag
        
        # Network output
        out = self.net(features)
        
        raw_diag = out[:, :self.n_diag]
        raw_lower = out[:, self.n_diag:]
        
        # Diagonal of L: positive via softplus, bounded
        L_diag = F.softplus(raw_diag)
        L_diag = torch.clamp(L_diag, 
                            min=np.sqrt(self.min_var), 
                            max=np.sqrt(self.max_var))
        
        # Lower triangle: bounded
        L_lower = torch.tanh(raw_lower) * L_diag[:, self.tril_idx[0]] * 0.5
        
        # Build L
        L = torch.zeros(B, self.nx, self.nx, device=device)
        L[:, range(self.nx), range(self.nx)] = L_diag
        if self.n_lower > 0:
            L[:, self.tril_idx[0], self.tril_idx[1]] = L_lower
        
        # P = L @ L^T (guaranteed SPD)
        P = L @ L.transpose(-1, -2)
        
        return P


# =============================================================================
# UKF CORE
# =============================================================================

class UKFCore:
    """UKF update equations."""
    
    def __init__(self, nx, nz, R, device, alpha=0.001, beta=2.0, kappa=0.0):
        self.nx = nx
        self.nz = nz
        self.device = device
        
        self.R = torch.as_tensor(R, dtype=torch.float32, device=device)
        if self.R.dim() == 2:
            self.R = self.R.unsqueeze(0)
        
        # UKF weights
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
    
    def _sigma_points(self, x, P):
        B = x.shape[0]
        
        P_reg = P + 1e-4 * torch.eye(self.nx, device=self.device)
        try:
            L = torch.linalg.cholesky(P_reg)
        except:
            eigvals, eigvecs = torch.linalg.eigh(P_reg)
            eigvals = torch.clamp(eigvals, min=1e-4)
            L = eigvecs @ torch.diag_embed(torch.sqrt(eigvals))
        
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
        
        eigvals, eigvecs = torch.linalg.eigh(P_post)
        eigvals = torch.clamp(eigvals, min=1e-4)
        P_post = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        
        return x_post, P_post, y, S


# =============================================================================
# LOSS
# =============================================================================

def nll_loss(y, S, clamp_max=100.0):
    """NLL with clamping for stability."""
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clamp(eigvals, min=1e-6)
    
    log_det = torch.sum(torch.log(eigvals), dim=-1)
    log_det = torch.clamp(log_det, min=-50, max=50)
    
    y_rot = (eigvecs.transpose(-1, -2) @ y.unsqueeze(-1)).squeeze(-1)
    mahal = torch.sum(y_rot**2 / eigvals, dim=-1)
    mahal = torch.clamp(mahal, max=clamp_max)
    
    return torch.mean(log_det + mahal)


# =============================================================================
# TRAINING - CURRICULUM LEARNING
# =============================================================================

def train_unsupervised_neural_kf(
    z_train, R, h_fn,
    x0, P0,
    nx, nz,
    epochs=500,
    lr=1e-3,
    device='cpu',
    checkpoint_path='neural_kf_unsupervised.pth',
    curriculum=True,
    verbose=True
):
    """
    Train fully unsupervised Neural KF.
    
    Uses curriculum learning: start with short sequences, gradually increase.
    """
    device = torch.device(device)
    
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    B, T_max, _ = z_train.shape
    
    if verbose:
        print(f"[Train] Fully Unsupervised Neural KF")
        print(f"[Train] B={B}, T_max={T_max}, nx={nx}, nz={nz}")
        print(f"[Train] Epochs={epochs}, LR={lr}")
        print(f"[Train] Curriculum learning: {curriculum}")
    
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_t = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    # Networks - no F matrix!
    mean_net = MeanNetUnsupervised(nx, hidden=128, layers=3).to(device)
    cov_net = CovNetUnsupervised(nx, hidden=128, layers=3).to(device)
    
    # UKF
    ukf = UKFCore(nx, nz, R, device)
    
    # Optimizer with weight decay for regularization
    params = list(mean_net.parameters()) + list(cov_net.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    
    # Learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6
    )
    
    history = {'loss': [], 'best_epoch': 0, 'T_curriculum': []}
    best_loss = float('inf')
    
    for ep in range(epochs):
        mean_net.train()
        cov_net.train()
        optimizer.zero_grad()
        
        # Curriculum: gradually increase sequence length
        if curriculum:
            # Start with T=3, increase every 50 epochs
            T = min(3 + (ep // 50) * 2, T_max)
        else:
            T = T_max
        
        history['T_curriculum'].append(T)
        
        # Random subsequence start for variety
        if T < T_max:
            max_start = T_max - T
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start_idx = 0
        
        # Initialize
        x = x0_t.unsqueeze(0).expand(B, -1).clone()
        P = P0_t.unsqueeze(0).expand(B, -1, -1).clone()
        
        total_loss = 0.0
        valid_steps = 0
        
        for t in range(T):
            t_actual = start_idx + t
            
            # Neural prediction (no F matrix!)
            x_pred = mean_net(x)
            P_pred = cov_net(P)
            
            # UKF update
            x, P, y, S = ukf.update(x_pred, P_pred, z_train[:, t_actual], h_fn)
            
            # NLL loss
            loss_t = nll_loss(y, S)
            
            if torch.isnan(loss_t) or torch.isinf(loss_t):
                continue
            
            total_loss = total_loss + loss_t
            valid_steps += 1
        
        if valid_steps == 0:
            continue
        
        loss = total_loss / valid_steps
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        
        # Skip if gradient explosion
        if grad_norm > 100:
            if verbose and ep % 50 == 0:
                print(f"[Train] Ep {ep}: gradient explosion ({grad_norm:.1f}), skipping")
            continue
        
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        history['loss'].append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            history['best_epoch'] = ep
            torch.save({
                'mean': mean_net.state_dict(),
                'cov': cov_net.state_dict(),
                'nx': nx,
                'nz': nz,
            }, checkpoint_path)
        
        if verbose and ep % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {ep:4d} | Loss: {loss_val:.4f} | Best: {best_loss:.4f} @ {history['best_epoch']} | T={T} | LR: {current_lr:.2e}")
    
    # Load best
    ckpt = torch.load(checkpoint_path, map_location=device)
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    if verbose:
        print(f"\n[Train] Done. Best loss: {best_loss:.4f} @ epoch {history['best_epoch']}")
    
    return mean_net, cov_net, history


# =============================================================================
# INFERENCE
# =============================================================================

class NeuralKFUnsupervised:
    """Fully unsupervised Neural KF for inference."""
    
    def __init__(self, mean_net, cov_net, R, device):
        self.mean_net = mean_net.to(device)
        self.cov_net = cov_net.to(device)
        self.device = device
        self.nx = mean_net.nx
        self.nz = R.shape[0]
        
        self.ukf = UKFCore(self.nx, self.nz, R, device)
    
    def run(self, z_seq, x0, P0, h_fn):
        """Run filter on sequence."""
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
        
        states = []
        covs = []
        
        with torch.no_grad():
            for t in range(T):
                x_pred = self.mean_net(x)
                P_pred = self.cov_net(P)
                x, P, _, _ = self.ukf.update(x_pred, P_pred, z_seq[:, t], h_fn)
                states.append(x.cpu().numpy())
                covs.append(P.cpu().numpy())
        
        return np.array(states).transpose(1, 0, 2), np.array(covs).transpose(1, 0, 2, 3)


def load_unsupervised_neural_kf(path, R, device):
    """Load trained unsupervised model."""
    ckpt = torch.load(path, map_location=device)
    nx = ckpt['nx']
    
    mean_net = MeanNetUnsupervised(nx).to(device)
    cov_net = CovNetUnsupervised(nx).to(device)
    
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    return NeuralKFUnsupervised(mean_net, cov_net, R, device)