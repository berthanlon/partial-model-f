# predict_nn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanNet(nn.Module):
    def __init__(self, nx, hidden=128, layers=3, state_scale=None):
        super().__init__()
        self.nx = nx
        if state_scale is None: state_scale = torch.ones(nx)
        self.register_buffer('state_scale', state_scale)
        
        modules = []
        prev = nx
        for _ in range(layers):
            modules.append(nn.Linear(prev, hidden))
            modules.append(nn.LayerNorm(hidden))
            modules.append(nn.Tanh())
            prev = hidden
        modules.append(nn.Linear(hidden, nx))
        self.net = nn.Sequential(*modules)
        
        # Init: Gain=1.0 for signal flow
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalize
        x_norm = x / self.state_scale
        
        # Network predicts NORMALIZED correction
        delta = self.net(x_norm)
        
        # Tanh Limit: +/- 50% of the state scale
        # Prevents "teleportation" to infinity
        delta = 0.5 * torch.tanh(delta) * self.state_scale
        
        return x + delta

class CovNetCholesky(nn.Module):
    def __init__(self, nx, hidden=128, layers=2, cov_scale=None):
        super().__init__()
        self.nx = nx
        if cov_scale is None: cov_scale = torch.ones(nx)
        self.register_buffer('cov_scale', cov_scale)
        self.register_buffer('log_cov_scale', torch.log(cov_scale + 1e-8))
        
        self.n_diag = nx
        self.n_lower = nx * (nx - 1) // 2
        input_size = nx + self.n_lower
        
        modules = []
        prev = input_size
        for _ in range(layers):
            modules.append(nn.Linear(prev, hidden))
            modules.append(nn.LayerNorm(hidden))
            modules.append(nn.Tanh())
            prev = hidden
        modules.append(nn.Linear(hidden, self.n_diag + self.n_lower))
        self.net = nn.Sequential(*modules)
        
        self.register_buffer('tril_idx', torch.tril_indices(nx, nx, offset=-1))
        self.register_buffer('diag_idx', torch.arange(nx))

        # Init Very Small: Start near Identity
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, P):
        B = P.shape[0]
        
        # Robust Feature Extraction
        d = torch.diagonal(P, dim1=-2, dim2=-1)
        log_d = torch.log(d.clamp(min=1e-6)) - self.log_cov_scale
        
        # Features must be finite
        feat = torch.nan_to_num(log_d, 0.0) 
        
        if self.n_lower > 0:
            L_lower = P[:, self.tril_idx[0], self.tril_idx[1]]
            # Normalized lower triangle features
            feat_L = L_lower / (torch.sqrt(self.cov_scale.sum()) + 1e-6)
            feat = torch.cat([feat, feat_L], dim=-1)
            
        out = self.net(feat)
        d_delta = out[:, :self.n_diag]
        l_delta = out[:, self.n_diag:]
        
        # UPDATE DIAGONALS (Log-Space Bounded)
        # 0.1 * Tanh limits update to small multiplicative steps
        log_L = 0.5 * (self.log_cov_scale + log_d) + 0.1 * torch.tanh(d_delta)
        L_diag = torch.exp(log_L)
        
        # Construct L
        L = torch.zeros(B, self.nx, self.nx, device=P.device)
        L[:, self.diag_idx, self.diag_idx] = L_diag
        
        if self.n_lower > 0:
            # Scale lower triangle by diagonal magnitude
            scale_fac = torch.sqrt(L_diag[:, self.tril_idx[0]] * L_diag[:, self.tril_idx[1]])
            L[:, self.tril_idx[0], self.tril_idx[1]] = 0.1 * torch.tanh(l_delta) * scale_fac
            
        # P = L @ L^T
        # Guaranteed Positive Definite by construction
        return L @ L.transpose(-1, -2)

def verify_spd(P, name="P", eps=1e-8):
    """Verify SPD."""
    sym_error = torch.abs(P - P.transpose(-1, -2)).max().item()
    
    try:
        eigenvalues = torch.linalg.eigvalsh(P)
        min_eig = eigenvalues.min().item()
        max_eig = eigenvalues.max().item()
        
        is_spd = (sym_error < eps) and (min_eig > -eps)
        
        print(f"[{name}] sym_err={sym_error:.2e}, eig=[{min_eig:.2e}, {max_eig:.2e}], SPD={is_spd}")
        
        return is_spd
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return False