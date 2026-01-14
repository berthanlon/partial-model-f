# kf_nets_v4.py
"""
Neural Predictor KF for inference - v4 (original scale).
"""
import torch
import numpy as np
from predict_nn import MeanNet, CovNetCholesky


def _sym(M):
    return 0.5 * (M + M.transpose(-1, -2))


def _project_spd(M, eps=1e-6):
    M = _sym(M)
    M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    
    try:
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals = torch.clamp(eigvals, min=eps)
        M = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
    except Exception:
        B, n, _ = M.shape
        diag = torch.diagonal(M, dim1=-2, dim2=-1).clamp(min=eps)
        M = torch.diag_embed(diag)
    
    return M


class NeuralPredictorKF:
    def __init__(self, nx, nz, R, mean_net, cov_net, device='cpu',
                 nis_thresh=9.21, y_min_clamp=None):
        self.nx = nx
        self.nz = nz
        self.device = torch.device(device)
        self.nis_thresh = nis_thresh
        self.y_min_clamp = y_min_clamp
        
        self.m = mean_net.to(self.device)
        self.Cnn = cov_net.to(self.device)
        
        R_tensor = torch.as_tensor(R, dtype=torch.float32, device=self.device)
        self.R = R_tensor.unsqueeze(0) if R_tensor.dim() == 2 else R_tensor
        
        # UKF params
        self.alpha = 0.001
        self.beta = 2.0
        self.kappa = 0.0
        self.lmbda = self.alpha**2 * (nx + self.kappa) - nx
        
        n_sigma = 2 * nx + 1
        self.n_sigma = n_sigma
        
        Wm = torch.zeros(n_sigma, device=self.device)
        Wc = torch.zeros(n_sigma, device=self.device)
        Wm[0] = self.lmbda / (nx + self.lmbda)
        Wc[0] = Wm[0] + (1 - self.alpha**2 + self.beta)
        Wm[1:] = 1.0 / (2.0 * (nx + self.lmbda))
        Wc[1:] = Wm[1:]
        
        self.Wm = Wm
        self.Wc = Wc
        self.gamma = np.sqrt(nx + self.lmbda)
        
        self.Q_floor = torch.eye(nx, device=self.device) * 0.01
    
    def _safe_cholesky(self, P):
        B = P.shape[0]
        P_reg = P + 1e-6 * torch.eye(self.nx, device=self.device).unsqueeze(0)
        try:
            return torch.linalg.cholesky(P_reg)
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(P_reg)
            eigvals = torch.clamp(eigvals, min=1e-6)
            return eigvecs @ torch.diag_embed(torch.sqrt(eigvals))
    
    def _generate_sigma_points(self, x, P):
        B = x.shape[0]
        L = self._safe_cholesky(P)
        
        sigma = torch.zeros(B, self.n_sigma, self.nx, device=self.device, dtype=x.dtype)
        sigma[:, 0, :] = x
        
        scaled_L = self.gamma * L
        for i in range(self.nx):
            sigma[:, i + 1, :] = x + scaled_L[:, :, i]
            sigma[:, self.nx + i + 1, :] = x - scaled_L[:, :, i]
        
        return sigma
    
    def step_nonlinear(self, x_prev, P_prev, z_k, h_fn):
        B = x_prev.shape[0]
        if z_k.ndim == 1:
            z_k = z_k.unsqueeze(0).expand(B, -1)
        
        # Neural prediction
        x_pred = self.m(x_prev)
        P_pred = self.Cnn(P_prev)
        P_pred = P_pred + self.Q_floor.unsqueeze(0)
        P_pred = _project_spd(P_pred)
        
        # UKF update
        sigma = self._generate_sigma_points(x_pred, P_pred)
        
        sigma_flat = sigma.reshape(B * self.n_sigma, self.nx)
        z_sigma_flat = h_fn(sigma_flat)
        z_sigma = z_sigma_flat.reshape(B, self.n_sigma, self.nz)
        
        z_hat = torch.sum(self.Wm.view(1, -1, 1) * z_sigma, dim=1)
        
        z_diff = z_sigma - z_hat.unsqueeze(1)
        S = torch.zeros(B, self.nz, self.nz, device=self.device, dtype=x_pred.dtype)
        for i in range(self.n_sigma):
            outer = z_diff[:, i, :].unsqueeze(-1) @ z_diff[:, i, :].unsqueeze(-2)
            S = S + self.Wc[i] * outer
        S = S + self.R.expand(B, -1, -1)
        S = _sym(S)
        S = S + 1e-6 * torch.eye(self.nz, device=self.device).unsqueeze(0)
        
        x_diff = sigma - x_pred.unsqueeze(1)
        C = torch.zeros(B, self.nx, self.nz, device=self.device, dtype=x_pred.dtype)
        for i in range(self.n_sigma):
            outer = x_diff[:, i, :].unsqueeze(-1) @ z_diff[:, i, :].unsqueeze(-2)
            C = C + self.Wc[i] * outer
        
        # Kalman gain
        S_inv = torch.linalg.inv(S)
        K = C @ S_inv
        
        y = z_k - z_hat
        
        # NIS gating
        if self.nis_thresh is not None:
            nis = (y.unsqueeze(1) @ S_inv @ y.unsqueeze(-1)).squeeze()
            gate = (nis < self.nis_thresh).float().view(-1, 1)
        else:
            gate = torch.ones(B, 1, device=self.device)
        
        x_post = x_pred + gate * (K @ y.unsqueeze(-1)).squeeze(-1)
        
        I = torch.eye(self.nx, device=self.device).unsqueeze(0)
        KC = K @ C.transpose(-1, -2)
        ImKC = I - gate.unsqueeze(-1) * KC
        P_post = ImKC @ P_pred @ ImKC.transpose(-1, -2)
        P_post = P_post + gate.unsqueeze(-1) * K @ self.R.expand(B, -1, -1) @ K.transpose(-1, -2)
        P_post = _project_spd(P_post)
        
        if self.y_min_clamp is not None:
            x_post = x_post.clone()
            x_post[:, 2] = torch.clamp(x_post[:, 2], min=self.y_min_clamp)
        
        return x_post, P_post, x_pred, P_pred, S, y, z_hat, None