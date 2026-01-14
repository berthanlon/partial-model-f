# metrics.py — XY-only RMSE + NIS utilities
from __future__ import annotations
import numpy as np
import torch

def _to_numpy(a):
    import torch
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

# ----------------------------- RMSE helpers ----------------------------------
def batch_rmse_states_xy(x_bt, gt_bt, reflect_y=False):
    """
    Per-dimension RMSE for x and y across (batch,time).
    Returns array of shape (2,) for [x, y].
    """
    x = _to_numpy(x_bt).copy()
    g = _to_numpy(gt_bt)
    if reflect_y:
        dot = np.sum(x[:, :, 2] * g[:, :, 2], axis=1, keepdims=True)
        s = np.sign(dot); s[s == 0] = 1.0
        x[:, :, 2] *= s
    x = x[:, :, (0, 2)]
    g = g[:, :, (0, 2)]
    return np.sqrt(np.mean((x - g)**2, axis=(0, 1)))

def rmse_time_across_sequences_xy(x_bt, gt_bt, reflect_y=False):
    """
    computeMSEsForSequences semantics for XY:
      rmse_T[t] = sqrt( mean_over_sequences( sum_k (x-g)^2 ) )
    Returns (rmse_T, overall_rmse).
    """
    x = _to_numpy(x_bt).copy()
    g = _to_numpy(gt_bt)
    if reflect_y:
        dot = np.sum(x[:, :, 2] * g[:, :, 2], axis=1, keepdims=True)
        s = np.sign(dot); s[s == 0] = 1.0
        x[:, :, 2] *= s
    x = x[:, :, (0, 2)]
    g = g[:, :, (0, 2)]
    se_sum = ((x - g)**2).sum(axis=2)    # (B,T)
    rmse_T = np.sqrt(se_sum.mean(axis=0))
    overall = float(np.sqrt(se_sum.mean()))
    return rmse_T, overall

def rmse_time_xy(x_bt, gt_bt, keep_idx=(0, 2)):
    """RMSE(t) per component; returns rmse_x_t, rmse_y_t."""
    x = np.asarray(x_bt)[:, :, keep_idx]
    g = np.asarray(gt_bt)[:, :, keep_idx]
    e = x - g                              # (B,T,2)
    rmse_tk = np.sqrt(np.mean(e**2, axis=0))  # (T,2)
    return rmse_tk[:, 0], rmse_tk[:, 1]

def total_rmse_time_xy(x_bt, gt_bt):
    """Compute RMSE(t) on combined XY: sqrt(mean_b[(ex^2+ey^2)])."""
    x = np.asarray(x_bt)[:, :, (0, 2)]
    g = np.asarray(gt_bt)[:, :, (0, 2)]
    se = (x - g) ** 2
    se_sum = se.sum(axis=2)               # (B,T)
    return np.sqrt(se_sum.mean(axis=0))   # (T,)

# ---------------------------- NLL / NIS utilities ----------------------------
def nll_decomposition(S_bt, innov_bt):
    S = _to_numpy(S_bt)      # (B,T,nz,nz)
    y = _to_numpy(innov_bt)  # (B,T,nz)
    nz = y.shape[-1]
    # log |S| and y^T S^{-1} y (solve via cholesky of S)
    logdetS = np.log(np.clip(np.linalg.det(S), 1e-12, None))
    mean_logdet = float(np.mean(logdetS))
    # Solve S * z = y  -> z = S^{-1} y
    z = np.linalg.solve(S, y[..., None]).squeeze(-1)  # (B,T,nz)
    nis = np.sum(y * z, axis=-1)                     # (B,T)
    mean_nis = float(np.mean(nis))
    return mean_logdet, mean_nis, nis

def nis_fraction_below(nis_bt, nz, alpha=0.95):
    """
    Fraction of NIS values below chi-square threshold (approx).
    """
    from scipy.stats import chi2
    thr = chi2.ppf(alpha, df=nz)
    return float(np.mean(nis_bt < thr))


def y_sign_from_gt(gt_bt: np.ndarray) -> np.ndarray:
    """
    Decide per-sequence sign for y based on the *ground truth*:
    s[b] = +1 if sum_t y_{b,t} >= 0, else -1.
    """
    s = np.where(np.sum(gt_bt[:, :, 2], axis=1) >= 0.0, 1.0, -1.0)[:, None]
    return s.astype(float)

def apply_y_sign(x_bt: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Apply sign s (B,1) to y and vy components of x_bt (B,T,nx).
    """
    x = np.array(x_bt, dtype=float, copy=True)
    if x.shape[-1] >= 3:
        x[:, :, 2] *= s
    if x.shape[-1] >= 4:
        x[:, :, 3] *= s
    return x

# Normalize UKF to (B,T,nx)
def _to_B_T_nx(a: np.ndarray, T_expected: int, nx_expected: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[1] == nx_expected and a.shape[2] == T_expected:  # (B,nx,T)
        return np.transpose(a, (0, 2, 1))
    if a.ndim == 2 and a.shape[0] == T_expected and a.shape[1] == nx_expected:   # (T,nx) single seq
        return a[None, ...]
    if a.ndim == 3 and a.shape[1] == T_expected and a.shape[2] == nx_expected:   # (B,T,nx)
        return a
    raise ValueError(
        f"Unexpected UKF output shape {a.shape}; expected (B,T,{nx_expected}) "
        f"or (B,{nx_expected},{T_expected})."
    )

# ---- KNet overlay (canonical subdirectory only) -----------------------------
def _to_B_T_nx_lenient(a: np.ndarray, T_expected: int, nx_expected: int) -> np.ndarray | None:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[1] == nx_expected and a.shape[2] == T_expected:  # (B,nx,T)
        return np.transpose(a, (0, 2, 1))
    if a.ndim == 3 and a.shape[1] == T_expected and a.shape[2] == nx_expected:  # (B,T,nx)
        return a
    if a.ndim == 2 and a.shape[0] == T_expected and a.shape[1] == nx_expected:  # (T,nx)
        return a[None, ...]
    print(f"[KNet] Unexpected shape {a.shape}. Expect (B,T,{nx_expected})/(B,{nx_expected},{T_expected}) or (T,{nx_expected}).")
    return None

# ---------------- RMSE overlays: Neural vs UKF vs (optional) KNet ------------
def _finite_or_none(a: np.ndarray, name: str):
    if not np.all(np.isfinite(a)):
        bad = np.where(~np.isfinite(a))[0][:5]
        print(f"[WARN] Non-finite values in {name}; sample bad idx:", bad)
        return None
    return a