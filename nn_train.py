"""
Neural Kalman Filter v23 - Simpler Architecture + Diagnostics

DIAGNOSIS OF v22 FAILURE:
========================
The training loss went UP, meaning the network couldn't even fit the training data.
The multi-step and consistency losses added instability without helping.

ROOT CAUSE ANALYSIS:
===================
1. The measurement model (range from 2 radars) weakly observes velocity
2. Gradient signal for learning velocity dynamics is very weak
3. Complex losses (multi-step, consistency) added noise without signal

NEW APPROACH:
============
1. Much simpler architecture - just learn the dynamics
2. Careful initialization 
3. Freeze covariance initially, focus on mean prediction
4. Add diagnostic tools to understand what's happening

KEY INSIGHT: Maybe the problem isn't the architecture but the LOSS.
The NLL loss for Kalman filtering is:
    L = log|S| + y^T S^{-1} y
where S = H P H^T + R and y = z - h(x_pred)

This loss doesn't directly reward good STATE predictions - it rewards good 
MEASUREMENT predictions. If the measurement model doesn't distinguish between
different velocities well, the network has no gradient signal to learn velocity.

Let's verify this hypothesis and try a simpler baseline first.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class StateScaler:
    def __init__(self, pos_scale=100.0, vel_scale=1.0, device='cpu'):
        self.device = device
        self.scale = torch.tensor([pos_scale, vel_scale, pos_scale, vel_scale], 
                                   dtype=torch.float32, device=device)
        self.inv_scale = 1.0 / self.scale
        
    def normalize(self, x_phys): 
        return x_phys * self.inv_scale
    
    def denormalize(self, x_norm): 
        return x_norm * self.scale
    
    def scale_cov(self, P):
        D_inv = self.inv_scale.view(-1, 1) * self.inv_scale.view(1, -1)
        return P * D_inv
    
    def unscale_cov(self, P_norm):
        D = self.scale.view(-1, 1) * self.scale.view(1, -1)
        return P_norm * D


class SimpleDynamicsNet(nn.Module):
    """
    Simplest possible dynamics: x' = x + f(x)
    where f is a small MLP.
    
    No bells and whistles - just learn the residual.
    """
    def __init__(self, nx, hidden=64):
        super().__init__()
        self.nx = nx
        
        self.net = nn.Sequential(
            nn.Linear(nx, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, nx)
        )
        
        # Initialize to output near-zero
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        self.net[-1].weight.data.mul_(0.01)
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return x + self.net(x)
    
    def get_delta(self, x):
        return self.net(x)


class FixedCovPredictor(nn.Module):
    """
    Covariance prediction with residual structure: P_{k|k-1} = P_{k-1|k-1} + ΔP_θ(P_{k-1|k-1})
    
    This prevents the covariance from collapsing to arbitrarily small values
    because the network can only ADD to the covariance, not shrink it arbitrarily.
    
    The network learns ΔP as L @ L^T (guaranteed SPD), so P_new = P_old + L @ L^T
    which is guaranteed to be at least as large as P_old in the PSD sense.
    
    This is a middle ground between:
    - Too restrictive: P + Q (constant Q)
    - Too flexible: General C_θ(P) (can collapse)
    """
    def __init__(self, nx, hidden=64):
        super().__init__()
        self.nx = nx
        self.n_tril = nx * (nx + 1) // 2  # Elements in lower triangular
        
        # Input: features of P (log diagonal + lower triangular)
        n_input = nx + nx * (nx - 1) // 2
        
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_tril)
        )
        
        # Initialize to output small values (small ΔP initially)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        
        # Bias diagonal to small positive values
        # softplus(-3) ≈ 0.05, so ΔP will have small diagonal initially
        self.net[-1].bias.data[:nx].fill_(-3.0)
        
        self.register_buffer('tril_indices', torch.tril_indices(nx, nx, offset=-1))
        self.register_buffer('diag_indices', torch.arange(nx))
    
    def _extract_features(self, P):
        """Extract features from covariance matrix."""
        B = P.shape[0]
        # Log of diagonal (variances)
        diag = torch.diagonal(P, dim1=1, dim2=2)
        log_diag = torch.log(diag.clamp(min=1e-8))
        # Lower triangular (covariances)
        lower = P[:, self.tril_indices[0], self.tril_indices[1]]
        return torch.cat([log_diag, lower], dim=1)
    
    def forward(self, P_old):
        B = P_old.shape[0]
        
        # Extract features from input covariance
        features = self._extract_features(P_old)
        
        # Predict Cholesky factor elements for ΔP
        L_elements = self.net(features)
        
        # Build lower triangular matrix L
        L = torch.zeros(B, self.nx, self.nx, device=P_old.device)
        
        # Diagonal: must be positive (use softplus)
        L[:, self.diag_indices, self.diag_indices] = F.softplus(L_elements[:, :self.nx]) + 1e-6
        
        # Off-diagonal
        L[:, self.tril_indices[0], self.tril_indices[1]] = L_elements[:, self.nx:]
        
        # ΔP = L @ L^T (guaranteed SPD)
        delta_P = L @ L.transpose(-1, -2)
        
        # RESIDUAL: P_new = P_old + ΔP
        # This guarantees P_new >= P_old in PSD sense (can't shrink!)
        P_new = P_old + delta_P
        
        return P_new
    
    def get_delta_P(self, P_old):
        """For diagnostics: return just the learned ΔP."""
        B = P_old.shape[0]
        features = self._extract_features(P_old)
        L_elements = self.net(features)
        
        L = torch.zeros(B, self.nx, self.nx, device=P_old.device)
        L[:, self.diag_indices, self.diag_indices] = F.softplus(L_elements[:, :self.nx]) + 1e-6
        L[:, self.tril_indices[0], self.tril_indices[1]] = L_elements[:, self.nx:]
        
        return L @ L.transpose(-1, -2)


class UKFCore:
    def __init__(self, nx, nz, R, device):
        self.nx = nx
        self.nz = nz
        self.device = device
        self.R = torch.as_tensor(R, dtype=torch.float32, device=device)
        if self.R.dim() == 2:
            self.R = self.R.unsqueeze(0)
        
        alpha, beta, kappa = 1.0, 2.0, 0.0
        lmbda = alpha**2 * (nx + kappa) - nx
        self.gamma = np.sqrt(nx + lmbda)
        self.n_sigma = 2 * nx + 1
        
        Wm = torch.zeros(self.n_sigma, device=device)
        Wc = torch.zeros(self.n_sigma, device=device)
        denom = nx + lmbda
        Wm[0] = lmbda / denom
        Wc[0] = Wm[0] + (1 - alpha**2 + beta)
        Wm[1:] = 0.5 / denom
        Wc[1:] = Wm[1:]
        self.Wm, self.Wc = Wm, Wc

    def _safe_cholesky(self, P):
        try:
            return torch.linalg.cholesky(P + 1e-5 * torch.eye(self.nx, device=self.device))
        except:
            eigvals, eigvecs = torch.linalg.eigh(P + 1e-5 * torch.eye(self.nx, device=self.device))
            eigvals = torch.clamp(eigvals, min=1e-5)
            return eigvecs @ torch.diag_embed(torch.sqrt(eigvals))

    def update(self, x_pred, P_pred, z, h_fn):
        B = x_pred.shape[0]
        L = self._safe_cholesky(P_pred)
        sigmas = [x_pred]
        for i in range(self.nx):
            sigmas.append(x_pred + self.gamma * L[:, :, i])
            sigmas.append(x_pred - self.gamma * L[:, :, i])
        sigma = torch.stack(sigmas, dim=1)
        
        z_sigma = h_fn(sigma.reshape(-1, self.nx)).reshape(B, self.n_sigma, self.nz)
        z_hat = (self.Wm.view(1, -1, 1) * z_sigma).sum(1)
        z_diff = z_sigma - z_hat.unsqueeze(1)
        x_diff = sigma - x_pred.unsqueeze(1)
        
        S = self.R.expand(B, -1, -1).clone()
        C = torch.zeros(B, self.nx, self.nz, device=self.device)
        
        for i in range(self.n_sigma):
            wd = self.Wc[i]
            S = S + wd * (z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
            C = C + wd * (x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
            
        try:
            K = torch.linalg.solve(S, C.transpose(1, 2)).transpose(1, 2)
        except:
            K = C @ torch.linalg.pinv(S)
            
        y = z - z_hat
        x_new = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_new = P_pred - K @ S @ K.transpose(-1, -2)
        P_new = 0.5 * (P_new + P_new.transpose(-1, -2))
        
        return x_new, P_new, y, S


class NeuralKFv23:
    def __init__(self, mean_net, cov_net, scaler, R, device):
        self.mean_net = mean_net
        self.cov_net = cov_net
        self.scaler = scaler
        self.ukf = UKFCore(4, 2, R, device)
        self.device = device
        
    def run(self, z_seq, x0, P0, h_fn):
        self.mean_net.eval()
        self.cov_net.eval()
        
        z_seq = torch.as_tensor(z_seq, dtype=torch.float32, device=self.device)
        if z_seq.dim() == 2:
            z_seq = z_seq.unsqueeze(0)
        B, T, _ = z_seq.shape
        
        x = torch.as_tensor(x0, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1).clone()
        P = torch.as_tensor(P0, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()
        
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


def train_v23(z_train, R, h_fn, x0, P0, nx, nz, 
              pos_scale=100.0, vel_scale=1.0,
              epochs=400, lr=1e-3, device='cpu',
              checkpoint_path=None, verbose=True, **kwargs):
    """
    Simple training with NLL loss only.
    Focus on stability and understanding what's learnable.
    """
    device = torch.device(device)
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    
    scaler = StateScaler(pos_scale, vel_scale, device)
    mean_net = SimpleDynamicsNet(nx).to(device)
    cov_net = FixedCovPredictor(nx, hidden=64).to(device)
    ukf = UKFCore(nx, nz, R, device)
    
    # Start with frozen covariance
    for p in cov_net.parameters():
        p.requires_grad = False
    
    optimizer = optim.Adam(mean_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, min_lr=1e-5)
    
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_t = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    if verbose:
        print(f"[Train] v23 Simple Baseline")
        print(f"  Phase 1 (ep 0-199): Train mean only, freeze cov")
        print(f"  Phase 2 (ep 200+): Train both")
    
    history = {'loss': [], 'best_epoch': 0, 'best_loss': float('inf')}
    best_state = None
    
    B, T, _ = z_train.shape
    
    for ep in range(epochs):
        # Phase 2: unfreeze covariance
        if ep == 200:
            if verbose:
                print(">>> Phase 2: Unfreezing covariance <<<")
            for p in cov_net.parameters():
                p.requires_grad = True
            optimizer = optim.Adam([
                {'params': mean_net.parameters(), 'lr': lr * 0.1},
                {'params': cov_net.parameters(), 'lr': lr * 0.1},
            ])
        
        mean_net.train()
        cov_net.train()
        optimizer.zero_grad()
        
        x = x0_t.unsqueeze(0).expand(B, -1).clone()
        P = P0_t.unsqueeze(0).expand(B, -1, -1).clone()
        
        total_loss = 0
        tbptt_len = 10
        chunk_loss = 0
        
        for t in range(T):
            x_norm = scaler.normalize(x)
            P_norm = scaler.scale_cov(P)
            
            x_pred_norm = mean_net(x_norm)
            P_pred_norm = cov_net(P_norm)
            
            x_pred = scaler.denormalize(x_pred_norm)
            P_pred = scaler.unscale_cov(P_pred_norm)
            
            x, P, y, S = ukf.update(x_pred, P_pred, z_train[:, t], h_fn)
            
            # NLL loss
            try:
                L_S = torch.linalg.cholesky(S + 1e-6 * torch.eye(nz, device=device))
                log_det = 2 * torch.sum(torch.log(torch.diagonal(L_S, dim1=-2, dim2=-1)), dim=-1)
                sol = torch.linalg.solve(L_S, y.unsqueeze(-1)).squeeze(-1)
                mahal = torch.sum(sol**2, dim=-1)
            except:
                mahal = torch.sum(y**2, dim=-1)
                log_det = torch.zeros_like(mahal) + 5.0
            
            loss_t = torch.mean(log_det + mahal)
            chunk_loss += loss_t
            
            if (t + 1) % tbptt_len == 0:
                avg_loss = chunk_loss / tbptt_len
                avg_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(mean_net.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(cov_net.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                x = x.detach()
                P = P.detach()
                total_loss += avg_loss.item()
                chunk_loss = 0
        
        history['loss'].append(total_loss)
        scheduler.step(total_loss)
        
        if total_loss < history['best_loss']:
            history['best_loss'] = total_loss
            history['best_epoch'] = ep
            best_state = {
                'mean_net': {k: v.cpu().clone() for k, v in mean_net.state_dict().items()},
                'cov_net': {k: v.cpu().clone() for k, v in cov_net.state_dict().items()},
            }
        
        if verbose and ep % 20 == 0:
            with torch.no_grad():
                # Test delta at a few points
                test_x = torch.tensor([
                    [1.0, 0.01, 0.5, 0.005],  # normalized typical state
                ], device=device, dtype=torch.float32)
                delta = mean_net.get_delta(test_x)[0].cpu().numpy()
                
                # Test covariance prediction - show the learned ΔP
                test_P = torch.eye(nx, device=device).unsqueeze(0) * 0.1
                delta_P = cov_net.get_delta_P(test_P)
                dP_diag = torch.diag(delta_P[0]).cpu().numpy()
            
            print(f"Ep {ep:3d} | Loss: {total_loss:.3f} | "
                  f"delta=[{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}, {delta[3]:.4f}] | "
                  f"ΔP_diag=[{dP_diag[0]:.4f}, {dP_diag[1]:.4f}, {dP_diag[2]:.4f}, {dP_diag[3]:.4f}]")
    
    # Load best
    if best_state is not None:
        mean_net.load_state_dict({k: v.to(device) for k, v in best_state['mean_net'].items()})
        cov_net.load_state_dict({k: v.to(device) for k, v in best_state['cov_net'].items()})
        if verbose:
            print(f"\nLoaded best model from epoch {history['best_epoch']}")
    
    if checkpoint_path:
        torch.save({
            'mean_net': mean_net.state_dict(),
            'cov_net': cov_net.state_dict(),
            'history': history,
        }, checkpoint_path)
    
    return mean_net, cov_net, scaler, history


def analyze_gradients(mean_net, cov_net, scaler, ukf, z_batch, x0, P0, h_fn, device):
    """
    Diagnostic: Analyze gradient magnitudes to understand what's learnable.
    """
    mean_net.train()
    
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_t = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    B = z_batch.shape[0]
    x = x0_t.unsqueeze(0).expand(B, -1).clone()
    P = P0_t.unsqueeze(0).expand(B, -1, -1).clone()
    
    # Single step
    x_norm = scaler.normalize(x)
    P_norm = scaler.scale_cov(P)
    
    x_pred_norm = mean_net(x_norm)
    P_pred_norm = cov_net(P_norm)
    
    x_pred = scaler.denormalize(x_pred_norm)
    P_pred = scaler.unscale_cov(P_pred_norm)
    
    x_new, P_new, y, S = ukf.update(x_pred, P_pred, z_batch[:, 0], h_fn)
    
    # Compute gradients w.r.t. each output component
    grads = {}
    for i, name in enumerate(['x', 'vx', 'y', 'vy']):
        mean_net.zero_grad()
        loss = x_pred[:, i].sum()
        loss.backward(retain_graph=True)
        
        total_grad = 0
        for p in mean_net.parameters():
            if p.grad is not None:
                total_grad += p.grad.abs().sum().item()
        grads[f'd_pred_{name}'] = total_grad
    
    # Gradient from NLL loss
    mean_net.zero_grad()
    try:
        L_S = torch.linalg.cholesky(S + 1e-6 * torch.eye(2, device=device))
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L_S, dim1=-2, dim2=-1)), dim=-1)
        sol = torch.linalg.solve(L_S, y.unsqueeze(-1)).squeeze(-1)
        mahal = torch.sum(sol**2, dim=-1)
        nll = torch.mean(log_det + mahal)
    except:
        nll = torch.mean(y**2)
    
    nll.backward()
    total_grad = 0
    for p in mean_net.parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().sum().item()
    grads['d_NLL'] = total_grad
    
    return grads


if __name__ == "__main__":
    print("Testing SimpleDynamicsNet...")
    net = SimpleDynamicsNet(4)
    x = torch.randn(5, 4)
    y = net(x)
    delta = net.get_delta(x)
    print(f"Max delta at init: {delta.abs().max().item():.6f}")
    
    print("\nTesting FixedCovPredictor...")
    cov = FixedCovPredictor(4)
    P = torch.eye(4).unsqueeze(0)
    P_new = cov(P)
    Q = cov.get_Q()
    print(f"Q diagonal: {Q.diag().detach().numpy()}")
    print(f"P_new diagonal: {P_new[0].diag().detach().numpy()}")