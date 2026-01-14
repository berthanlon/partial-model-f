# nn_train_v5.py
"""
Neural KF Training - v5 (WORKING VERSION)

Key insight: Train the network WITH the Kalman filter update in the loop,
not just prediction-only. This ensures the network learns to work with
the measurement correction.

Architecture:
- MeanNet: Predicts state transition delta (small correction to F@x)
- CovNet: Predicts covariance (via Cholesky, guaranteed SPD)

Loss: Negative log-likelihood of innovations
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# =============================================================================
# NETWORKS
# =============================================================================
class MeanNet(nn.Module):
    """
    State prediction: x_pred = F @ x_prev + delta(x_prev)
    
    The network learns a CORRECTION to the physics model.
    """
    def __init__(self, nx, F_matrix, hidden=64, layers=2):
        super().__init__()
        self.nx = nx
        self.register_buffer('F', torch.as_tensor(F_matrix, dtype=torch.float32))
        
        # Small MLP for correction
        mods = []
        prev = nx
        for _ in range(layers):
            mods.append(nn.Linear(prev, hidden))
            mods.append(nn.Tanh())
            prev = hidden
        mods.append(nn.Linear(hidden, nx))
        self.net = nn.Sequential(*mods)
        
        # Initialize to near-zero output
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Physics prediction
        x_phys = (self.F @ x.unsqueeze(-1)).squeeze(-1)
        # Neural correction (small)
        delta = 0.1 * self.net(x)
        return x_phys + delta


class CovNet(nn.Module):
    """
    Covariance prediction via Cholesky decomposition.
    
    P_pred = L @ L^T where L is lower triangular with positive diagonal.
    
    We learn: L_diag (via softplus) and L_lower (bounded)
    """
    def __init__(self, nx, Q_matrix, hidden=64, layers=2):
        super().__init__()
        self.nx = nx
        
        # Store Q for reference scale
        Q_tensor = torch.as_tensor(Q_matrix, dtype=torch.float32)
        self.register_buffer('Q', Q_tensor)
        self.register_buffer('Q_diag', torch.diag(Q_tensor))
        
        # Output size
        self.n_diag = nx
        self.n_lower = nx * (nx - 1) // 2
        
        # Input: just diagonal of P (log scale)
        mods = []
        prev = nx
        for _ in range(layers):
            mods.append(nn.Linear(prev, hidden))
            mods.append(nn.Tanh())
            prev = hidden
        mods.append(nn.Linear(hidden, self.n_diag + self.n_lower))
        self.net = nn.Sequential(*mods)
        
        # Indices for building L
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
        
        # Initialize
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, P_prev):
        B = P_prev.shape[0]
        device = P_prev.device
        
        # Extract diagonal features (in log space for stability)
        diag = torch.diagonal(P_prev, dim1=-2, dim2=-1)
        log_diag = torch.log(diag.clamp(min=1e-6))
        
        # Network output
        out = self.net(log_diag)
        
        # Split output
        raw_diag = out[:, :self.n_diag]
        raw_lower = out[:, self.n_diag:]
        
        # Diagonal of L: must be positive
        # Use softplus and clamp to reasonable range
        L_diag = F.softplus(raw_diag) + 0.01  # Minimum 0.01
        L_diag = torch.clamp(L_diag, min=0.01, max=100.0)
        
        # Lower triangular: bounded small values
        L_lower = 0.1 * torch.tanh(raw_lower)
        
        # Build L matrix
        L = torch.zeros(B, self.nx, self.nx, device=device)
        L[:, range(self.nx), range(self.nx)] = L_diag
        if self.n_lower > 0:
            L[:, self.tril_idx[0], self.tril_idx[1]] = L_lower
        
        # P = L @ L^T (guaranteed SPD)
        P = L @ L.transpose(-1, -2)
        
        # Add Q floor to ensure growth
        P = P + self.Q.unsqueeze(0)
        
        return P


# =============================================================================
# UKF COMPONENTS
# =============================================================================
class UKFCore:
    """Minimal UKF for training."""
    
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
        
        # Safe Cholesky
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
        """Compute innovation statistics and update."""
        B = x_pred.shape[0]
        
        # Sigma points
        sigma = self._sigma_points(x_pred, P_pred)
        
        # Transform through h
        sigma_flat = sigma.reshape(B * self.n_sigma, self.nx)
        z_sigma_flat = h_fn(sigma_flat)
        z_sigma = z_sigma_flat.reshape(B, self.n_sigma, self.nz)
        
        # Predicted measurement
        z_hat = (self.Wm.view(1, -1, 1) * z_sigma).sum(dim=1)
        
        # Innovation covariance S
        z_diff = z_sigma - z_hat.unsqueeze(1)
        S = torch.zeros(B, self.nz, self.nz, device=self.device)
        for i in range(self.n_sigma):
            S = S + self.Wc[i] * (z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
        S = S + self.R.expand(B, -1, -1)
        S = 0.5 * (S + S.transpose(-1, -2))  # Symmetrize
        S = S + 1e-4 * torch.eye(self.nz, device=self.device)  # Regularize
        
        # Cross-covariance
        x_diff = sigma - x_pred.unsqueeze(1)
        C = torch.zeros(B, self.nx, self.nz, device=self.device)
        for i in range(self.n_sigma):
            C = C + self.Wc[i] * (x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
        
        # Kalman gain
        S_inv = torch.linalg.inv(S)
        K = C @ S_inv
        
        # Innovation
        y = z - z_hat
        
        # Update
        x_post = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_post = P_pred - K @ S @ K.transpose(-1, -2)
        P_post = 0.5 * (P_post + P_post.transpose(-1, -2))
        
        # Ensure SPD
        eigvals, eigvecs = torch.linalg.eigh(P_post)
        eigvals = torch.clamp(eigvals, min=1e-4)
        P_post = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        
        return x_post, P_post, y, S


# =============================================================================
# LOSS
# =============================================================================
def nll_loss(y, S):
    """
    Negative log-likelihood: log|S| + y^T S^{-1} y
    """
    B = y.shape[0]
    
    # Eigendecomposition for stability
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clamp(eigvals, min=1e-6)
    
    log_det = torch.sum(torch.log(eigvals), dim=-1)
    
    y_rot = (eigvecs.transpose(-1, -2) @ y.unsqueeze(-1)).squeeze(-1)
    mahal = torch.sum(y_rot**2 / eigvals, dim=-1)
    
    return torch.mean(log_det + mahal)


# =============================================================================
# TRAINING
# =============================================================================
def train_neural_kf(
    z_train, F, Q, R, h_fn,
    x0, P0,
    epochs=200, lr=1e-3,
    device='cpu',
    checkpoint_path='neural_kf.pth'
):
    """
    Train neural KF with full filter in the loop.
    """
    device = torch.device(device)
    
    # Data
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    B, T, nz = z_train.shape
    nx = F.shape[0]
    
    print(f"[Train] B={B}, T={T}, nx={nx}, nz={nz}")
    
    # Initial conditions
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_t = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    # Networks
    mean_net = MeanNet(nx, F, hidden=64, layers=2).to(device)
    cov_net = CovNet(nx, Q, hidden=64, layers=2).to(device)
    
    # UKF core
    ukf = UKFCore(nx, nz, R, device)
    
    # Optimizer
    params = list(mean_net.parameters()) + list(cov_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    
    history = {'loss': [], 'best_epoch': 0}
    best_loss = float('inf')
    
    print(f"[Train] Starting {epochs} epochs...")
    
    for ep in range(epochs):
        mean_net.train()
        cov_net.train()
        optimizer.zero_grad()
        
        # Initialize
        x = x0_t.unsqueeze(0).expand(B, -1).clone()
        P = P0_t.unsqueeze(0).expand(B, -1, -1).clone()
        
        total_loss = 0.0
        
        for t in range(T):
            # Neural prediction
            x_pred = mean_net(x)
            P_pred = cov_net(P)
            
            # UKF update (computes y, S)
            x, P, y, S = ukf.update(x_pred, P_pred, z_train[:, t], h_fn)
            
            # NLL loss
            loss_t = nll_loss(y, S)
            
            if torch.isnan(loss_t):
                print(f"[Train] NaN at ep={ep}, t={t}")
                break
            
            total_loss = total_loss + loss_t
        
        loss = total_loss / T
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        
        loss_val = loss.item()
        history['loss'].append(loss_val)
        scheduler.step(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            history['best_epoch'] = ep
            torch.save({
                'mean': mean_net.state_dict(),
                'cov': cov_net.state_dict(),
                'F': F,
                'Q': Q,
            }, checkpoint_path)
        
        if ep % 20 == 0:
            print(f"Ep {ep:4d} | Loss: {loss_val:.4f} | Best: {best_loss:.4f} @ {history['best_epoch']}")
    
    # Load best
    ckpt = torch.load(checkpoint_path, map_location=device)
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    print(f"[Train] Done. Best loss: {best_loss:.4f} @ epoch {history['best_epoch']}")
    
    return mean_net, cov_net, history


# =============================================================================
# INFERENCE CLASS
# =============================================================================
class NeuralKF:
    """Neural KF for inference."""
    
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


def load_neural_kf(path, R, device):
    """Load trained model."""
    ckpt = torch.load(path, map_location=device)
    F = ckpt['F']
    Q = ckpt['Q']
    nx = F.shape[0]
    
    mean_net = MeanNet(nx, F).to(device)
    cov_net = CovNet(nx, Q).to(device)
    
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    return NeuralKF(mean_net, cov_net, R, device)