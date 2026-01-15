# nn_train_v10.py
"""
FULLY UNSUPERVISED Neural Kalman Filter - v10

Key fix: RESIDUAL architecture that CANNOT collapse to constant output.

The network predicts a DELTA, not the full next state:
    x_next = x + delta_net(x)

This ensures:
1. Identity is the default (if delta_net outputs 0)
2. Network can't ignore the input
3. Gradients flow directly through the identity path
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# STATE SCALER
# =============================================================================
class StateScaler:
    def __init__(self, pos_scale=100.0, vel_scale=5.0, device='cpu'):
        self.device = device
        self.scale = torch.tensor([pos_scale, vel_scale, pos_scale, vel_scale], 
                                   dtype=torch.float32, device=device)
        self.inv_scale = 1.0 / self.scale
        
    def normalize(self, x):
        return x * self.inv_scale
    
    def denormalize(self, x_norm):
        return x_norm * self.scale
    
    def scale_cov(self, P):
        D_inv = torch.diag(self.inv_scale)
        return D_inv @ P @ D_inv
    
    def unscale_cov(self, P_norm):
        D = torch.diag(self.scale)
        return D @ P_norm @ D


# =============================================================================
# MEAN NETWORK - RESIDUAL ARCHITECTURE
# =============================================================================
class MeanNetResidual(nn.Module):
    """
    Predicts state transition as: x_next = x + delta(x)
    
    The delta network is initialized to output the CV dynamics correction:
    - delta_x = vx * dt_normalized  (where dt_normalized = dt * vel_scale / pos_scale)
    - delta_vx = 0
    - delta_y = vy * dt_normalized
    - delta_vy = 0
    
    For dt=0.5, pos_scale=100, vel_scale=5:
        dt_normalized = 0.5 * 5 / 100 = 0.025
    """
    def __init__(self, nx, hidden=64, dt_normalized=0.025):
        super().__init__()
        self.nx = nx
        self.dt_norm = dt_normalized
        
        # Delta network - predicts correction to identity
        self.delta_net = nn.Sequential(
            nn.Linear(nx, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, nx)
        )
        
        # Initialize delta_net to output approximately CV dynamics
        self._init_cv_dynamics()
    
    def _init_cv_dynamics(self):
        """
        Initialize to approximate constant-velocity dynamics.
        
        We need the network to compute: delta = [vx*dt, 0, vy*dt, 0]
        
        With architecture: Linear -> Tanh -> Linear -> Tanh -> Linear
        
        Strategy: Make the first layer pass velocity through, 
        and the last layer scale it by dt_norm.
        """
        # First, initialize all to small random values
        for m in self.delta_net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        # Now set up the "velocity passthrough" path
        first_layer = self.delta_net[0]  # Linear(4, hidden)
        last_layer = self.delta_net[-1]   # Linear(hidden, 4)
        hidden_size = first_layer.out_features
        
        with torch.no_grad():
            # First layer: make first two hidden units copy vx and vy
            # hidden[0] = vx (input[1])
            # hidden[1] = vy (input[3])
            first_layer.weight.zero_()
            first_layer.bias.zero_()
            first_layer.weight[0, 1] = 1.0  # hidden[0] = vx
            first_layer.weight[1, 3] = 1.0  # hidden[1] = vy
            
            # Last layer: map hidden[0] -> dx, hidden[1] -> dy
            # After tanh, for inputs in [-1, 1], tanh(x) ≈ x
            last_layer.weight.zero_()
            last_layer.bias.zero_()
            last_layer.weight[0, 0] = self.dt_norm  # dx = tanh(vx) * dt_norm ≈ vx * dt_norm
            last_layer.weight[2, 1] = self.dt_norm  # dy = tanh(vy) * dt_norm ≈ vy * dt_norm
    
    def forward(self, x_norm):
        """
        x_norm: (B, nx) normalized state [x, vx, y, vy]
        returns: (B, nx) predicted next normalized state
        """
        # RESIDUAL: x_next = x + delta
        delta = self.delta_net(x_norm)
        return x_norm + delta


# =============================================================================
# COVARIANCE NETWORK
# =============================================================================
class CovNetSimple(nn.Module):
    def __init__(self, nx, init_std=0.1):
        super().__init__()
        self.nx = nx
        self.log_L_diag = nn.Parameter(torch.full((nx,), np.log(init_std)))
        n_lower = nx * (nx - 1) // 2
        self.L_lower = nn.Parameter(torch.zeros(n_lower))
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
    
    def forward(self, P_prev_norm):
        B = P_prev_norm.shape[0]
        device = P_prev_norm.device
        
        L_diag = torch.exp(self.log_L_diag)
        L_diag = torch.clamp(L_diag, min=0.001, max=10.0)
        
        L = torch.zeros(B, self.nx, self.nx, device=device)
        L[:, range(self.nx), range(self.nx)] = L_diag
        
        if self.L_lower.numel() > 0:
            scale = torch.sqrt(L_diag[self.tril_idx[0]] * L_diag[self.tril_idx[1]])
            L_low = torch.tanh(self.L_lower) * 0.5 * scale
            L[:, self.tril_idx[0], self.tril_idx[1]] = L_low
        
        P = L @ L.transpose(-1, -2)
        return P + 1e-6 * torch.eye(self.nx, device=device)


# =============================================================================
# UKF CORE
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
    
    def update(self, x_pred, P_pred, z, h_fn):
        B = x_pred.shape[0]
        
        L = self._safe_cholesky(P_pred)
        sigma = torch.zeros(B, self.n_sigma, self.nx, device=self.device)
        sigma[:, 0] = x_pred
        for i in range(self.nx):
            sigma[:, i+1] = x_pred + self.gamma * L[:, :, i]
            sigma[:, self.nx+i+1] = x_pred - self.gamma * L[:, :, i]
        
        z_sigma = h_fn(sigma.reshape(-1, self.nx)).reshape(B, self.n_sigma, self.nz)
        z_hat = (self.Wm.view(1, -1, 1) * z_sigma).sum(dim=1)
        
        z_diff = z_sigma - z_hat.unsqueeze(1)
        S = self.R.expand(B, -1, -1).clone()
        for i in range(self.n_sigma):
            S = S + self.Wc[i] * (z_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
        S = 0.5 * (S + S.transpose(-1, -2))
        S = S + 1e-4 * torch.eye(self.nz, device=self.device)
        
        x_diff = sigma - x_pred.unsqueeze(1)
        C = torch.zeros(B, self.nx, self.nz, device=self.device)
        for i in range(self.n_sigma):
            C = C + self.Wc[i] * (x_diff[:, i].unsqueeze(-1) @ z_diff[:, i].unsqueeze(-2))
        
        try:
            S_inv = torch.linalg.inv(S)
        except:
            S_inv = torch.linalg.pinv(S)
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
def nll_loss(y, S):
    try:
        L = torch.linalg.cholesky(S)
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-6), dim=-1)
        z = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False).squeeze(-1)
        mahal = torch.sum(z ** 2, dim=-1)
    except:
        eigvals = torch.linalg.eigvalsh(S)
        eigvals = torch.clamp(eigvals, min=1e-6)
        log_det = torch.sum(torch.log(eigvals), dim=-1)
        mahal = torch.sum(y ** 2, dim=-1) / eigvals.mean(dim=-1)
    
    log_det = torch.clamp(log_det, -50, 50)
    mahal = torch.clamp(mahal, 0, 100)
    
    return torch.mean(log_det + mahal)


# =============================================================================
# TRAINING
# =============================================================================
def train_v10(
    z_train, R, h_fn,
    x0, P0,
    nx, nz,
    dt=0.5,
    pos_scale=100.0,
    vel_scale=5.0,
    epochs=1000,
    lr=1e-3,
    device='cpu',
    checkpoint_path='neural_kf_v10.pth',
    tbptt_len=5,
    verbose=True
):
    device = torch.device(device)
    
    z_train = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    B, T_max, _ = z_train.shape
    
    # Compute normalized dt
    dt_normalized = dt * vel_scale / pos_scale
    
    scaler = StateScaler(pos_scale, vel_scale, device)
    
    if verbose:
        print(f"[Train] v10 - Residual Architecture")
        print(f"[Train] dt={dt}, dt_normalized={dt_normalized:.4f}")
        print(f"[Train] B={B}, T_max={T_max}")
    
    x0_phys = torch.as_tensor(x0, dtype=torch.float32, device=device)
    P0_phys = torch.as_tensor(P0, dtype=torch.float32, device=device)
    
    mean_net = MeanNetResidual(nx, hidden=64, dt_normalized=dt_normalized).to(device)
    cov_net = CovNetSimple(nx, init_std=0.1).to(device)
    
    ukf = UKFCore(nx, nz, R, device)
    
    params = list(mean_net.parameters()) + list(cov_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {'loss': [], 'best_epoch': 0}
    best_loss = float('inf')
    
    # Verify initialization
    if verbose:
        mean_net.eval()
        with torch.no_grad():
            test_in = torch.tensor([[1.0, 0.2, 0.5, 0.1]], device=device)
            test_out = mean_net(test_in)
            expected = test_in.clone()
            expected[0, 0] += test_in[0, 1] * dt_normalized
            expected[0, 2] += test_in[0, 3] * dt_normalized
            print(f"[Init Check] Input:    {test_in[0].cpu().numpy()}")
            print(f"[Init Check] Output:   {test_out[0].cpu().numpy()}")
            print(f"[Init Check] Expected: {expected[0].cpu().numpy()}")
        mean_net.train()
    
    for ep in range(epochs):
        mean_net.train()
        cov_net.train()
        
        T = min(5 + ep // 20, T_max)
        
        batch_idx = torch.randperm(B)[:min(512, B)]
        z_batch = z_train[batch_idx]
        B_batch = len(batch_idx)
        
        x_phys = x0_phys.unsqueeze(0).expand(B_batch, -1).clone()
        P_phys = P0_phys.unsqueeze(0).expand(B_batch, -1, -1).clone()
        
        total_loss = 0.0
        n_seg = 0
        
        for seg_start in range(0, T, tbptt_len):
            seg_end = min(seg_start + tbptt_len, T)
            
            optimizer.zero_grad()
            
            x_phys = x_phys.detach()
            P_phys = P_phys.detach()
            
            seg_loss = torch.tensor(0.0, device=device)
            valid = 0
            
            for t in range(seg_start, seg_end):
                # Normalize
                x_norm = scaler.normalize(x_phys)
                P_norm = scaler.scale_cov(P_phys)
                
                # Predict in normalized space
                x_pred_norm = mean_net(x_norm)
                P_pred_norm = cov_net(P_norm)
                
                # Denormalize
                x_pred_phys = scaler.denormalize(x_pred_norm)
                P_pred_phys = scaler.unscale_cov(P_pred_norm)
                
                # UKF update
                x_phys, P_phys, y, S = ukf.update(x_pred_phys, P_pred_phys, z_batch[:, t], h_fn)
                
                loss_t = nll_loss(y, S)
                
                if not (torch.isnan(loss_t) or torch.isinf(loss_t)):
                    seg_loss = seg_loss + loss_t
                    valid += 1
            
            if valid > 0 and seg_loss.requires_grad:
                seg_loss = seg_loss / valid
                seg_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
                total_loss += seg_loss.item()
                n_seg += 1
        
        scheduler.step()
        
        if n_seg > 0:
            avg_loss = total_loss / n_seg
            history['loss'].append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                history['best_epoch'] = ep
                torch.save({
                    'mean': mean_net.state_dict(),
                    'cov': cov_net.state_dict(),
                    'nx': nx,
                    'dt': dt,
                    'pos_scale': pos_scale,
                    'vel_scale': vel_scale,
                }, checkpoint_path)
            
            if verbose and ep % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"Ep {ep:4d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} @ {history['best_epoch']} | T={T} | LR={lr_now:.2e}")
    
    # Load best
    ckpt = torch.load(checkpoint_path, map_location=device)
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    if verbose:
        print(f"\n[Train] Done. Best: {best_loss:.4f} @ epoch {history['best_epoch']}")
    
    return mean_net, cov_net, scaler, history


# =============================================================================
# INFERENCE
# =============================================================================
class NeuralKFv10:
    def __init__(self, mean_net, cov_net, scaler, R, device):
        self.mean_net = mean_net.to(device)
        self.cov_net = cov_net.to(device)
        self.scaler = scaler
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
        
        x_phys = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        if x_phys.dim() == 1:
            x_phys = x_phys.unsqueeze(0).expand(B, -1)
        
        P_phys = torch.as_tensor(P0, dtype=torch.float32, device=self.device)
        if P_phys.dim() == 2:
            P_phys = P_phys.unsqueeze(0).expand(B, -1, -1)
        
        states, covs = [], []
        
        with torch.no_grad():
            for t in range(T):
                x_norm = self.scaler.normalize(x_phys)
                P_norm = self.scaler.scale_cov(P_phys)
                
                x_pred_norm = self.mean_net(x_norm)
                P_pred_norm = self.cov_net(P_norm)
                
                x_pred_phys = self.scaler.denormalize(x_pred_norm)
                P_pred_phys = self.scaler.unscale_cov(P_pred_norm)
                
                x_phys, P_phys, _, _ = self.ukf.update(x_pred_phys, P_pred_phys, z_seq[:, t], h_fn)
                
                states.append(x_phys.cpu().numpy())
                covs.append(P_phys.cpu().numpy())
        
        return np.array(states).transpose(1, 0, 2), np.array(covs).transpose(1, 0, 2, 3)


def load_v10(path, R, device):
    ckpt = torch.load(path, map_location=device)
    nx = ckpt['nx']
    dt = ckpt.get('dt', 0.5)
    pos_scale = ckpt.get('pos_scale', 100.0)
    vel_scale = ckpt.get('vel_scale', 5.0)
    dt_normalized = dt * vel_scale / pos_scale
    
    mean_net = MeanNetResidual(nx, dt_normalized=dt_normalized).to(device)
    cov_net = CovNetSimple(nx).to(device)
    scaler = StateScaler(pos_scale, vel_scale, device)
    
    mean_net.load_state_dict(ckpt['mean'])
    cov_net.load_state_dict(ckpt['cov'])
    
    return NeuralKFv10(mean_net, cov_net, scaler, R, device)