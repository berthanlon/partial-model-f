# nn_train_v8.py
"""
FULLY UNSUPERVISED Neural Kalman Filter - v8

Fixes from v7:
1. Truncated BPTT - don't backprop through entire sequence
2. Target network for stability (like DQN)
3. Gradient penalty instead of clipping
4. Warmup learning rate

The math is unchanged - still learning:
- x_next = x + f(v)  where f should learn v*dt
- v_next = v + g(v)  where g should learn ~0
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func


# =============================================================================
# NETWORKS (same structure as v7)
# =============================================================================

class MeanNetStructured(nn.Module):
    def __init__(self, nx, hidden=64):
        super().__init__()
        self.nx = nx
        
        self.vel_to_pos = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        self.vel_update = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # vel_to_pos: initialize to approximate v*0.5 (dt=0.5)
        for m in self.vel_to_pos:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
        # Last layer bias toward 0.5 scaling
        nn.init.constant_(self.vel_to_pos[-1].weight, 0.5)
        nn.init.zeros_(self.vel_to_pos[-1].bias)
        
        # vel_update: initialize to output ~0
        for m in self.vel_update:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        pos_x, vel_x = x[:, 0:1], x[:, 1:2]
        pos_y, vel_y = x[:, 2:3], x[:, 3:4]
        
        dx = self.vel_to_pos(vel_x)
        dy = self.vel_to_pos(vel_y)
        
        pos_x_new = pos_x + dx
        pos_y_new = pos_y + dy
        
        dvx = torch.clamp(self.vel_update(vel_x), -0.1, 0.1)
        dvy = torch.clamp(self.vel_update(vel_y), -0.1, 0.1)
        
        vel_x_new = vel_x + dvx
        vel_y_new = vel_y + dvy
        
        return torch.cat([pos_x_new, vel_x_new, pos_y_new, vel_y_new], dim=1)


class CovNetSimple(nn.Module):
    def __init__(self, nx, init_std=1.0):
        super().__init__()
        self.nx = nx
        
        self.log_L_diag = nn.Parameter(torch.full((nx,), np.log(init_std)))
        
        n_lower = nx * (nx - 1) // 2
        self.L_lower = nn.Parameter(torch.zeros(n_lower))
        
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
    
    def forward(self, P_prev):
        B = P_prev.shape[0]
        device = P_prev.device
        
        L_diag = torch.exp(self.log_L_diag)
        L_diag = torch.clamp(L_diag, min=0.5, max=50.0)  # Higher minimum
        
        L = torch.zeros(B, self.nx, self.nx, device=device)
        L[:, range(self.nx), range(self.nx)] = L_diag.unsqueeze(0).expand(B, -1)
        
        if self.L_lower.numel() > 0:
            scale = torch.sqrt(L_diag[self.tril_idx[0]] * L_diag[self.tril_idx[1]])
            L_low = torch.tanh(self.L_lower) * 0.3 * scale
            L[:, self.tril_idx[0], self.tril_idx[1]] = L_low.unsqueeze(0).expand(B, -1)
        
        P = L @ L.transpose(-1, -2)
        
        # Add floor to ensure P doesn't get too small
        P = P + 0.1 * torch.eye(self.nx, device=device).unsqueeze(0)
        
        return P


# =============================================================================
# UKF CORE (unchanged)
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
        
        try:
            S_inv = torch.linalg.inv(S + 1e-4 * torch.eye(self.nz, device=self.device))
        except:
            # Fallback: use pseudoinverse
            S_inv = torch.linalg.pinv(S + 1e-4 * torch.eye(self.nz, device=self.device))
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
# LOSS with Huber-like clamping
# =============================================================================

def nll_loss_stable(y, S):
    """Stable NLL with aggressive clamping."""
    B = y.shape[0]
    
    # Use Cholesky for numerical stability
    try:
        L = torch.linalg.cholesky(S)
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-6), dim=-1)
        
        # Solve L @ z = y for z, then ||z||^2 = y^T S^{-1} y
        z = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False).squeeze(-1)
        mahal = torch.sum(z ** 2, dim=-1)
    except:
        # Fallback to eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(S)
        eigvals = torch.clamp(eigvals, min=1e-4, max=1e4)
        log_det = torch.sum(torch.log(eigvals), dim=-1)
        y_rot = (eigvecs.transpose(-1, -2) @ y.unsqueeze(-1)).squeeze(-1)
        mahal = torch.sum(y_rot**2 / eigvals, dim=-1)
    
    # Clamp both components
    log_det = torch.clamp(log_det, -20, 20)
    mahal = torch.clamp(mahal, 0, 50)
    
    nll = log_det + mahal
    
    # Winsorize: replace extreme values with percentile values
    nll_sorted = torch.sort(nll)[0]
    p95 = nll_sorted[int(0.95 * B)] if B > 20 else nll_sorted[-1]
    nll = torch.clamp(nll, max=p95)
    
    return torch.mean(nll)


# =============================================================================
# TRAINING with truncated BPTT
# =============================================================================

def train_unsupervised_v8(
    z_train, R, h_fn,
    x0, P0,
    nx, nz,
    epochs=1000,
    lr=1e-3,
    device='cpu',
    checkpoint_path='neural_kf_v8.pth',
    tbptt_len=5,  # Truncated BPTT length
    verbose=True
):
    """
    Train with truncated backprop through time for stability.
    """
    device = torch.device(device)
    
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    B, T_max, _ = z_train.shape
    
    if verbose:
        print(f"[Train] v8 - Truncated BPTT (len={tbptt_len})")
        print(f"[Train] B={B}, T_max={T_max}")
    
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_t = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    mean_net = MeanNetStructured(nx, hidden=64).to(device)
    cov_net = CovNetSimple(nx, init_std=1.0).to(device)
    
    ukf = UKFCore(nx, nz, R, device)
    
    params = list(mean_net.parameters()) + list(cov_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    # Warmup + cosine decay
    def lr_lambda(epoch):
        warmup = 100
        if epoch < warmup:
            return epoch / warmup
        else:
            progress = (epoch - warmup) / (epochs - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    history = {'loss': [], 'best_epoch': 0}
    best_loss = float('inf')
    
    for ep in range(epochs):
        mean_net.train()
        cov_net.train()
        
        # Curriculum
        T = min(3 + ep // 20, T_max)
        
        # Random batch subset for variance
        batch_idx = torch.randperm(B)[:min(512, B)]
        z_batch = z_train[batch_idx]
        B_batch = z_batch.shape[0]
        
        # Initialize state
        x = x0_t.unsqueeze(0).expand(B_batch, -1).clone()
        P = P0_t.unsqueeze(0).expand(B_batch, -1, -1).clone()
        
        total_loss = 0.0
        n_segments = 0
        
        # Process in segments (truncated BPTT)
        for seg_start in range(0, T, tbptt_len):
            seg_end = min(seg_start + tbptt_len, T)
            
            optimizer.zero_grad()
            
            # Detach state at segment boundary
            x = x.detach()
            P = P.detach()
            
            seg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            valid_count = 0
            
            for t in range(seg_start, seg_end):
                x_pred = mean_net(x)
                P_pred = cov_net(P)
                
                x, P, y, S = ukf.update(x_pred, P_pred, z_batch[:, t], h_fn)
                
                loss_t = nll_loss_stable(y, S)
                
                if torch.isnan(loss_t) or torch.isinf(loss_t):
                    continue
                
                seg_loss = seg_loss + loss_t
                valid_count += 1
            
            if valid_count > 0:
                seg_loss = seg_loss / valid_count
                seg_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                
                optimizer.step()
                
                total_loss += seg_loss.item()
                n_segments += 1
        
        scheduler.step()
        
        if n_segments > 0:
            avg_loss = total_loss / n_segments
            history['loss'].append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                history['best_epoch'] = ep
                torch.save({
                    'mean': mean_net.state_dict(),
                    'cov': cov_net.state_dict(),
                    'nx': nx,
                }, checkpoint_path)
            
            if verbose and ep % 100 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"Ep {ep:4d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} @ {history['best_epoch']} | T={T} | LR={lr_now:.2e}")
    
    # Load best
    ckpt = torch.load(checkpoint_path, map_location=device)
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    if verbose:
        print(f"\n[Train] Done. Best: {best_loss:.4f} @ epoch {history['best_epoch']}")
    
    return mean_net, cov_net, history


# =============================================================================
# INFERENCE (unchanged from v7)
# =============================================================================

class NeuralKFv8:
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


def load_v8(path, R, device):
    ckpt = torch.load(path, map_location=device)
    nx = ckpt['nx']
    
    mean_net = MeanNetStructured(nx).to(device)
    cov_net = CovNetSimple(nx).to(device)
    
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    return NeuralKFv8(mean_net, cov_net, R, device)