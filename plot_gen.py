# plot_gen.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def _to_numpy(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)

def plot_state_timeseries(x_est: np.ndarray, P_est: np.ndarray, gt: np.ndarray, save_path: str):
    t = np.arange(x_est.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    mask = np.isfinite(x_est).all(axis=1)
    
    # Position X and Velocity VX
    axes[0,0].plot(t, gt[:,0], label="GT x", color="green")
    axes[0,0].plot(t[mask], x_est[mask,0], label="Neural Est x", color="blue")
    axes[0,0].set_ylabel("x"); axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(t, gt[:,1], label="GT vx", color="green")
    axes[0,1].plot(t[mask], x_est[mask,1], label="Neural Est vx", color="blue")
    axes[0,1].set_ylabel("vx"); axes[0,1].legend(); axes[0,1].grid(True)

    # Position Y and Velocity VY
    axes[1,0].plot(t, gt[:,2], label="GT y", color="green")
    axes[1,0].plot(t[mask], x_est[mask,2], label="Neural Est y", color="blue")
    axes[1,0].set_ylabel("y"); axes[1,0].grid(True)

    axes[1,1].plot(t, gt[:,3], label="GT vy", color="green")
    axes[1,1].plot(t[mask], x_est[mask,3], label="Neural Est vy", color="blue")
    axes[1,1].set_ylabel("vy"); axes[1,1].grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

# plot_gen.py
import os
import numpy as np
import matplotlib.pyplot as plt

def _to_numpy(a):
    import torch
    if isinstance(a, torch.Tensor): return a.detach().cpu().numpy()
    return np.asarray(a)

def plot_xy_trajectory(x_est, gt, save_path, radars=None):
    plt.figure(figsize=(6,5))
    x_est, gt = _to_numpy(x_est), _to_numpy(gt)
    
    # Plot Ground Truth
    plt.plot(gt[:,0], gt[:,2], 'g-', label="Ground Truth", linewidth=2, zorder=1)
    
    # Plot Neural Estimate 
    plt.scatter(x_est[:,0], x_est[:,2], color='blue', s=10, label="Neural", alpha=0.5, zorder=2)
    
    # LOCK VIEWPORT: Center on Ground Truth 
    cx, cy = gt[:,0].mean(), gt[:,2].mean()
    plt.xlim(cx - 150, cx + 150)
    plt.ylim(cy - 150, cy + 150)
    
    if radars:
        for r in radars: plt.plot(r[0], r[1], 'r^', markersize=8)

    plt.grid(True); plt.legend(); plt.title("XY Path (Fixed Scale)")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path); plt.close()

def plot_all_test_sequences(x_test, gt_test, output_dir, radars=None):
    """Generates XY plots and prints a health report for NaNs."""
    os.makedirs(output_dir, exist_ok=True)
    x_test = _to_numpy(x_test)
    gt_test = _to_numpy(gt_test)
    
    B, T, D = x_test.shape
    
    # --- Health Report ---
    nan_mask = np.isnan(x_test)
    inf_mask = np.isinf(x_test)
    total_nans = np.sum(nan_mask)
    total_infs = np.sum(inf_mask)
    
    print("-" * 30)
    print(f"DEBUG HEALTH REPORT:")
    print(f"Total NaNs in Neural Est: {total_nans}")
    print(f"Total Infs in Neural Est: {total_infs}")
    
    dead_seqs = np.any(nan_mask.reshape(B, -1), axis=1)
    print(f"Sequences with NaNs: {np.sum(dead_seqs)} out of {B}")
    
    if total_nans > 0:
        first_nan_idx = np.where(nan_mask)[1][0] if total_nans > 0 else "N/A"
        print(f"First NaN appeared at time step: {first_nan_idx}")
    print("-" * 30)

    for i in range(B):
        path = os.path.join(output_dir, f"seq_{i}_trajectory.png")
        # Only plot if there's at least some valid data
        if not np.all(nan_mask[i]):
            plot_xy_trajectory(x_test[i], gt_test[i], path, radars=radars)
        else:
            # Create a 'Failure' placeholder image to know which ones died
            plt.figure()
            plt.text(0.5, 0.5, f"Sequence {i} CRASHED (All NaNs)", ha='center')
            plt.savefig(path)
            plt.close()

def plot_total_rmse_vs_time_xy(neural, gt, ukf=None, save_path="", **kwargs):
    neural, gt = _to_numpy(neural), _to_numpy(gt)
    t = np.arange(neural.shape[1])
    
    # Calculate RMSE for X (idx 0) and Y (idx 2)
    err_n = (neural[:, :, [0,2]] - gt[:, :, [0,2]])**2
    mse_n = np.nanmean(err_n.sum(axis=2), axis=0) 
    rmse_n = np.sqrt(mse_n)

    plt.figure(figsize=(8, 4))
    plt.plot(t, rmse_n, label="Neural (Mean RMSE)", linewidth=2.5, color="blue")
    
    if ukf is not None:
        ukf = _to_numpy(ukf)
        err_u = (ukf[:, :, [0,2]] - gt[:, :, [0,2]])**2
        mse_u = np.nanmean(err_u.sum(axis=2), axis=0)
        plt.plot(t, np.sqrt(mse_u), label="UKF (Mean RMSE)", color="orange", linestyle="--")
        
    plt.title("Position RMSE vs Time")
    plt.xlabel("Time Step"); plt.ylabel("RMSE (m)"); plt.grid(True); plt.legend()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()

plot_rmse_vs_time = plot_total_rmse_vs_time_xy

def plot_est_vs_gt_seq(x_est, gt, seq, save_path, **kwargs):
    plot_state_timeseries(_to_numpy(x_est)[seq], None, _to_numpy(gt)[seq], save_path)

def save_states_csv(x_est, gt, save_path):
    T = x_est.shape[0]
    data = np.zeros((T, 5))
    data[:, 0] = np.arange(T)
    data[:, 1:3] = gt[:, [0,2]]
    data[:, 3:5] = x_est[:, [0,2]]
    np.savetxt(save_path, data, header="t,gt_x,gt_y,est_x,est_y", delimiter=",", comments="")