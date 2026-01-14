# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import torch

import parameters as P
from parameters import h, h_torch, H_jac_torch, RADAR_BASELINE
from system_model import SystemModel


# -----------------------------------------------------------------------------
# Batched filter rollout (robust triangulated init) — supports both EKF and UKF
# Returns 5 numpy arrays: (xs, Ps, zhat, S_all, y_all)
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_filter_batched(
    kf_obj,
    z_bt,
    h_fn=None,
    H_jac_fn=None,  # kept for backward compatibility, but ignored in UKF mode
    *,
    init_mode: str = "triangulate",        # "triangulate" | "bias" | "zero"
    estimate_vel_from_second: bool = False,
    dt: float = 1.0,
    init_bias: Optional[Dict[int, float]] = None,
    P_diag_override: Optional[Dict[int, float]] = None,
):
    """
    Vectorised filter rollout with robust triangulated initialisation.
    Works with both EKF (Jacobian-based) and UKF (sigma-point-based) updates.
      - Tries both sensor orders and ±y, picks best match to first measurement
      - Optional velocity estimation from second timestep
      - Supports initial bias and covariance override
    Returns: (xs, Ps, zhat, S_all, y_all) as numpy arrays
    """
    h = h_fn or h_torch  # Use provided h or default torch version
    # H_jac_fn is no longer used in UKF mode — kept only for API compatibility

    device = kf_obj.device
    z = torch.as_tensor(z_bt, dtype=torch.float32, device=device)  # (B, T, nz)
    B, T, nz = z.shape
    nx = kf_obj.nx

    def _triangulate_pair(rr1, rr2, b):
        x0 = (rr1 * rr1 - rr2 * rr2 + b * b) / (2.0 * b)
        y2 = torch.clamp(rr1 * rr1 - x0 * x0, min=0.0)
        yabs = torch.sqrt(y2)
        return x0, yabs

    def _robust_xy_from_z(zt: torch.Tensor, b: float):
        """Pick per-batch best (x,y) among 4 candidates (two orders × ±y)."""
        r1a, r2a = zt[:, 0], zt[:, 1]
        r1b, r2b = zt[:, 1], zt[:, 0]  # swapped

        x0a, ya = _triangulate_pair(r1a, r2a, b)
        x0b, yb = _triangulate_pair(r1b, r2b, b)

        # Candidates (B,4)
        x0_cands = torch.stack([x0a, x0a, x0b, x0b], dim=1)   # (B,4)
        y0_cands = torch.stack([ya, -ya, yb, -yb], dim=1)     # (B,4)

        # Candidate states (B,4,nx)
        x_init = torch.zeros(B, 4, nx, device=zt.device)
        x_init[:, :, 0] = x0_cands
        x_init[:, :, 2] = y0_cands

        # Predicted measurement for each candidate
        zhat = h(x_init.reshape(B * 4, nx)).reshape(B, 4, 2)  # (B,4,2)
        err = torch.linalg.norm(zhat - zt.unsqueeze(1), dim=2)  # (B,4)

        best_idx = torch.argmin(err, dim=1)  # (B,)
        x_best = torch.gather(x0_cands, 1, best_idx.unsqueeze(1)).squeeze(1)
        y_best = torch.gather(y0_cands, 1, best_idx.unsqueeze(1)).squeeze(1)
        return x_best, y_best

    # ---- Initial state ----
    x = torch.zeros(B, nx, device=device)
    if init_mode.lower() == "triangulate":
        b = float(RADAR_BASELINE)
        x0_best, y0_best = _robust_xy_from_z(z[:, 0, :], b)
        x[:, 0] = x0_best
        x[:, 2] = y0_best

        if estimate_vel_from_second and T >= 2:
            x1_best, y1_best = _robust_xy_from_z(z[:, 1, :], b)
            x[:, 1] = (x1_best - x0_best) / dt  # x velocity
            x[:, 3] = (y1_best - y0_best) / dt  # y velocity

    elif init_mode.lower() == "bias":
        pass  # start at zeros; bias applied below

    elif init_mode.lower() == "zero":
        pass  # already zero

    else:
        raise ValueError(f"Unknown init_mode={init_mode}")

    # Optional bias after chosen init
    if init_bias:
        for idx, val in init_bias.items():
            x[:, int(idx)] = float(val)

    # ---- Initial covariance P0 ----
    P0 = torch.eye(nx, device=device)
    if P_diag_override:
        for idx, var in P_diag_override.items():
            P0[int(idx), int(idx)] = float(var)
    P = P0.unsqueeze(0).expand(B, -1, -1).contiguous()

    # ---- Pre-allocate outputs ----
    xs    = torch.empty(B, T, nx,     device=device)
    Ps    = torch.empty(B, T, nx, nx, device=device)
    zhat  = torch.empty(B, T, nz,     device=device)
    S_all = torch.empty(B, T, nz, nz, device=device)
    y_all = torch.empty(B, T, nz,     device=device)

    # ---- Filter loop (now works with UKF step_nonlinear) ----
    for t in range(T):
        # UKF version of step_nonlinear does NOT use Jacobian → only pass h
        x, P, xpred, Ppred, S, y, zh, _ = kf_obj.step_nonlinear(x, P, z[:, t, :], h)

        xs[:, t, :]       = x
        Ps[:, t, :, :]    = P
        zhat[:, t, :]     = zh
        S_all[:, t, :, :] = S
        y_all[:, t, :]    = y

    # ---- Return as numpy arrays ----
    return (
        xs.cpu().numpy(),
        Ps.cpu().numpy(),
        zhat.cpu().numpy(),
        S_all.cpu().numpy(),
        y_all.cpu().numpy(),
    )